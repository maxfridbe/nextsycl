// ns-ocr-dump: llama.cpp's multimodal path (mtmd) as nextsycl's parity reference for OCR models (DeepSeek-OCR /
// Unlimited-OCR). The image goes through mtmd's preprocessing and vision graph, the prompt through the decoder, then
// N tokens are generated greedily (no repetition rule). Every graph tensor - the vision graphs' and the prompt pass's -
// whose name matches NS_DUMP_RE is written as float32 to <dir>/<name>.f32 (a name seen again: <name>#2, #3, ... in
// the order computed), its shape to <dir>/index.tsv (name, type, ne0..ne3, ne0 innermost); each vision graph's input
// image as inp_raw[#n].f32 (the preprocessed pixels the first convolution reads: [batch][3][H][W], rows of tiles as
// a batch of tiles). Token ids: <dir>/tokens.txt (each chunk: "text"
// and its ids, or "image N" and its positions), the generated ids: <dir>/gen.txt.
//
//   cp reference/llama-dump/ns-ocr-dump.cpp <llama.cpp>/tools/mtmd/   (see reference/llama-dump/README.md)
//   NS_DUMP_DIR=/tmp/ocr NS_DUMP_RE='^(sam_layer_out-[0-9]+|sam_output|layer_out-[0-9]+|dsocr_output|l_out-[0-9]+)$' \
//     llama-ns-ocr-dump -m Unlimited-OCR-Q8_0.gguf --mmproj mmproj-Unlimited-OCR-F16.gguf --image page.png \
//     -p "document parsing." -n 64 --device none -ngl 0
#include "arg.h"
#include "common.h"
#include "ggml-backend.h"
#include "llama.h"
#include "log.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <regex>
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

static void write_tensor(dump_state* st, const ggml_tensor* t, const std::string& base) {
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
    int k = ++st->seen[base];
    std::string name = k == 1 ? base : base + "#" + std::to_string(k);
    std::ofstream(st->dir + "/" + name + ".f32", std::ios::binary).write((const char*) f.data(), n * sizeof(float));
    fprintf(st->index, "%s\t%s\t%lld\t%lld\t%lld\t%lld\n", name.c_str(), ggml_type_name(t->type), (long long) t->ne[0],
            (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3]);
    fflush(st->index);
    st->written++;
}

static bool dump_cb(struct ggml_tensor* t, bool ask, void* user) {
    auto* st = (dump_state*) user;
    if (!st->on) return ask ? false : true;
    // a vision graph's first convolution: its source is the preprocessed image (or, for rows of tiles, the image
    // reshaped into a batch of tiles - followed back through the reshapes to inp_raw)
    bool raw_src = false;
    if (t->op == GGML_OP_IM2COL && t->src[1]) {
        const ggml_tensor* s = t->src[1];
        for (int i = 0; i < 8 && s; ++i) {
            if (strcmp(s->name, "inp_raw") == 0) { raw_src = true; break; }
            s = s->src[0];
        }
    }
    const bool named = std::regex_search(t->name, st->re);
    if (!named && !raw_src) return ask ? false : true;
    if (ask) return true;  // yes: call again once it is computed
    if (raw_src) write_tensor(st, t->src[1], "inp_raw");
    if (!named) return true;
    if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_BF16) return true;
    if (!ggml_is_contiguous(t)) {
        LOG_WRN("ns-ocr-dump: %s is not contiguous, skipped\n", t->name);
        return true;
    }
    write_tensor(st, t, t->name);
    return true;
}

int main(int argc, char** argv) {
    std::setlocale(LC_NUMERIC, "C");
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MTMD)) return 1;
    if (params.mmproj.path.empty() || params.image.empty()) {
        LOG_ERR("ns-ocr-dump: --mmproj and --image are needed\n");
        return 1;
    }

    dump_state st;
    st.dir = getenv("NS_DUMP_DIR") ? getenv("NS_DUMP_DIR") : "ns-ocr-dump";
    const char* re = getenv("NS_DUMP_RE");
    st.re = std::regex(re ? re : "^(sam_layer_out-[0-9]+|sam_output|layer_out-[0-9]+|dsocr_output|inp_embd|l_out-[0-9]+|result_norm|result_output)$");
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
    auto* lctx = init->context();
    if (!model || !lctx) {
        LOG_ERR("ns-ocr-dump: failed to load the model\n");
        return 1;
    }
    const llama_vocab* vocab = llama_model_get_vocab(model);

    mtmd_context_params mp = mtmd_context_params_default();
    mp.use_gpu = false;
    mp.n_threads = params.cpuparams.n_threads;
    mp.print_timings = true;
    mp.warmup = false;
    mp.cb_eval = dump_cb;
    mp.cb_eval_user_data = &st;
    mtmd_context* mctx = mtmd_init_from_file(params.mmproj.path.c_str(), model, mp);
    if (!mctx) {
        LOG_ERR("ns-ocr-dump: failed to load the projector\n");
        return 1;
    }

    std::vector<mtmd_bitmap*> bitmaps;
    std::string text;
    for (const auto& path : params.image) {
        mtmd_helper_bitmap_wrapper b = mtmd_helper_bitmap_init_from_file(mctx, path.c_str(), false, mtmd_helper_init_opt_default());
        if (!b.bitmap) {
            LOG_ERR("ns-ocr-dump: cannot read %s\n", path.c_str());
            return 1;
        }
        bitmaps.push_back(b.bitmap);
    }
    // the prompt places the picture itself when it holds the marker (a chat template: olmOCR's text comes first);
    // otherwise the pictures go first
    const std::string marker = mtmd_get_marker(mctx);
    if (params.prompt.find(marker) == std::string::npos)
        for (size_t i = 0; i < bitmaps.size(); ++i) text += marker;
    text += params.prompt;
    mtmd_input_text in{text.c_str(), text.size(), /*add_special*/ true, /*parse_special*/ true};
    mtmd_input_chunks* chunks = mtmd_input_chunks_init();
    std::vector<const mtmd_bitmap*> cb(bitmaps.begin(), bitmaps.end());
    if (mtmd_tokenize(mctx, chunks, &in, cb.data(), cb.size()) != 0) {
        LOG_ERR("ns-ocr-dump: tokenize failed\n");
        return 1;
    }
    {
        std::ofstream tf(st.dir + "/tokens.txt");
        for (size_t i = 0; i < mtmd_input_chunks_size(chunks); ++i) {
            const mtmd_input_chunk* c = mtmd_input_chunks_get(chunks, i);
            if (mtmd_input_chunk_get_type(c) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                size_t n = 0;
                const llama_token* t = mtmd_input_chunk_get_tokens_text(c, &n);
                tf << "text";
                for (size_t j = 0; j < n; ++j) tf << " " << t[j];
                tf << "\n";
            } else {
                tf << "image " << mtmd_input_chunk_get_n_tokens(c) << "\n";
            }
        }
    }
    llama_pos n_past = 0;
    if (mtmd_helper_eval_chunks(mctx, lctx, chunks, 0, 0, params.n_batch, true, &n_past) != 0) {
        LOG_ERR("ns-ocr-dump: eval failed\n");
        return 1;
    }
    st.on = false;  // the generated tokens' passes are not dumped
    const int nv = llama_vocab_n_tokens(vocab);
    std::ofstream gf(st.dir + "/gen.txt");
    std::string out;
    for (int i = 0; i < params.n_predict; ++i) {
        const float* logits = llama_get_logits_ith(lctx, -1);
        int best = 0;
        for (int j = 1; j < nv; ++j)
            if (logits[j] > logits[best]) best = j;
        gf << best << "\n";
        if (llama_vocab_is_eog(vocab, best)) break;
        out += common_token_to_piece(lctx, best);
        llama_token tok = best;
        if (llama_decode(lctx, llama_batch_get_one(&tok, 1))) {
            LOG_ERR("ns-ocr-dump: decode failed\n");
            return 1;
        }
    }
    std::ofstream(st.dir + "/gen-text.txt") << out;
    fclose(st.index);
    LOG_INF("ns-ocr-dump: %d tensors written to %s; %lld prompt positions\n%s\n", st.written, st.dir.c_str(),
            (long long) n_past, out.c_str());
    mtmd_input_chunks_free(chunks);
    for (auto* b : bitmaps) mtmd_bitmap_free(b);
    mtmd_free(mctx);
    llama_backend_free();
    return 0;
}
