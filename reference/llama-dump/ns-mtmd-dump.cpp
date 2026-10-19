// ns-mtmd-dump: llama.cpp's mtmd as nextsycl's parity reference for a model that reads pictures. The prompt (its text
// with mtmd's media marker <__media__> where each --image goes) is tokenized by mtmd and read; written to NS_DUMP_DIR:
//   image-<k>.f32    each picture's rows from the vision tower and projector [n_tokens][n_embd] float32
//   image-<k>.txt    its grid: n_tokens nx ny n_pos
//   tokens.txt       the text chunks' ids, a picture as "image <k> <n_tokens>"
//   logits.f32       the last prompt position's logits
//   next.txt         the greedy continuation's ids (NS_DUMP_GEN tokens, default 48), one a line
//
//   cp reference/llama-dump/ns-mtmd-dump.cpp <llama.cpp>/tools/mtmd/   (and the CMake lines in README.md)
//   NS_DUMP_DIR=/tmp/vdump NS_DUMP_PROMPT=prompt.txt llama-ns-mtmd-dump -m model.gguf --mmproj mmproj.gguf --image pic.png \
//     -p x --device none -ngl 0      (prompt.txt: e.g. "<|im_start|>user\n<__media__>What is in this picture?<|im_end|>\n...")
#include "arg.h"
#include "common.h"
#include "llama.h"
#include "log.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    std::setlocale(LC_NUMERIC, "C");
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MTMD)) return 1;
    const std::string dir = getenv("NS_DUMP_DIR") ? getenv("NS_DUMP_DIR") : "ns-mtmd-dump";
    const int n_gen = getenv("NS_DUMP_GEN") ? atoi(getenv("NS_DUMP_GEN")) : 48;
    if (system(("mkdir -p '" + dir + "'").c_str()) != 0) return 1;

    llama_backend_init();
    params.warmup = false;
    auto init = common_init_from_params(params);
    llama_model* model = init->model();
    llama_context* lctx = init->context();
    if (!model || !lctx) {
        LOG_ERR("ns-mtmd-dump: the model does not load\n");
        return 1;
    }
    mtmd_context_params mp = mtmd_context_params_default();
    mp.use_gpu = false;
    mp.n_threads = params.cpuparams.n_threads;
    mp.warmup = false;
    mtmd_context* mctx = mtmd_init_from_file(params.mmproj.path.c_str(), model, mp);
    if (!mctx) {
        LOG_ERR("ns-mtmd-dump: the mmproj file does not load\n");
        return 1;
    }
    std::vector<mtmd_bitmap*> bitmaps;
    for (auto& f : params.image) {
        auto w = mtmd_helper_bitmap_init_from_file(mctx, f.c_str(), false, mtmd_helper_init_opt_default());
        if (!w.bitmap) {
            LOG_ERR("ns-mtmd-dump: %s does not read\n", f.c_str());
            return 1;
        }
        bitmaps.push_back(w.bitmap);
    }
    // NS_DUMP_PROMPT: the prompt from this file as it is (-p and -f lose a trailing newline: "<think>\n" would end the
    // prompt without it)
    if (const char* pf = getenv("NS_DUMP_PROMPT")) {
        std::ifstream in(pf, std::ios::binary);
        params.prompt.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }
    mtmd_input_text text{params.prompt.c_str(), params.prompt.size(), false, true};
    mtmd_input_chunks* chunks = mtmd_input_chunks_init();
    if (mtmd_tokenize(mctx, chunks, &text, (const mtmd_bitmap**) bitmaps.data(), bitmaps.size()) != 0) {
        LOG_ERR("ns-mtmd-dump: tokenize failed\n");
        return 1;
    }
    const int n_embd = llama_model_n_embd_inp(model);
    std::ofstream tf(dir + "/tokens.txt");
    int k = 0;
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks); ++i) {
        const mtmd_input_chunk* c = mtmd_input_chunks_get(chunks, i);
        if (mtmd_input_chunk_get_type(c) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t n = 0;
            const llama_token* t = mtmd_input_chunk_get_tokens_text(c, &n);
            for (size_t j = 0; j < n; ++j) tf << t[j] << "\n";
            continue;
        }
        const size_t n = mtmd_input_chunk_get_n_tokens(c);
        if (mtmd_encode_chunk(mctx, c) != 0) {
            LOG_ERR("ns-mtmd-dump: encode failed\n");
            return 1;
        }
        const float* e = mtmd_get_output_embd(mctx);
        std::ofstream(dir + "/image-" + std::to_string(k) + ".f32", std::ios::binary).write((const char*) e, n * n_embd * sizeof(float));
        const mtmd_image_tokens* it = mtmd_input_chunk_get_tokens_image(c);
        auto last = mtmd_image_tokens_get_decoder_pos(it, 0, n - 1);
        std::ofstream(dir + "/image-" + std::to_string(k) + ".txt") << n << " " << last.x + 1 << " " << last.y + 1 << " " << mtmd_input_chunk_get_n_pos(c) << "\n";
        tf << "image " << k << " " << n << "\n";
        LOG_INF("ns-mtmd-dump: picture %d: %zu positions (%u x %u)\n", k, n, last.x + 1, last.y + 1);
        ++k;
    }
    tf.close();
    llama_pos n_past = 0;
    if (mtmd_helper_eval_chunks(mctx, lctx, chunks, 0, 0, params.n_batch, true, &n_past) != 0) {
        LOG_ERR("ns-mtmd-dump: eval failed\n");
        return 1;
    }
    const llama_vocab* vocab = llama_model_get_vocab(model);
    const int nv = llama_vocab_n_tokens(vocab);
    const float* logits = llama_get_logits_ith(lctx, -1);
    std::ofstream(dir + "/logits.f32", std::ios::binary).write((const char*) logits, nv * sizeof(float));
    std::ofstream nf(dir + "/next.txt");
    std::string out;
    llama_batch b = llama_batch_init(1, 0, 1);
    for (int g = 0; g < n_gen; ++g) {
        const float* l = llama_get_logits_ith(lctx, -1);
        int best = 0;
        for (int i = 1; i < nv; ++i)
            if (l[i] > l[best]) best = i;
        nf << best << "\n";
        if (llama_vocab_is_eog(vocab, best)) break;
        out += common_token_to_piece(lctx, best);
        b.n_tokens = 1;
        b.token[0] = best;
        b.pos[0] = n_past++;
        b.n_seq_id[0] = 1;
        b.seq_id[0][0] = 0;
        b.logits[0] = 1;
        if (llama_decode(lctx, b) != 0) {
            LOG_ERR("ns-mtmd-dump: decode failed\n");
            return 1;
        }
    }
    LOG_INF("ns-mtmd-dump: %d pictures; the greedy continuation: %s\n", k, out.c_str());
    printf("%s\n", out.c_str());
    mtmd_input_chunks_free(chunks);
    mtmd_free(mctx);
    llama_backend_free();
    return 0;
}
