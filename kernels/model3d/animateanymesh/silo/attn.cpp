// attn.cpp: AnimateAnyMesh's attention as float32 oneDNN products (silo.h). The engine's own kernel (aam.cpp) gives a
// work-group to every query row and re-reads all keys for each: 85% of a flow step. Here the scores of a block of
// query rows are one batched product over (sequence, head) - the inputs read in place through strides -, a softmax
// pass, and a second product writes the weighted values straight into the output's rows.
#include "ns.h"
#include "ns_internal.hpp"
#include "silo.h"

#include <sycl/sycl.hpp>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>

#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

namespace {
using dnnl::memory;

struct Dnn {
    dnnl::engine eng;
    dnnl::stream st;
    std::map<std::vector<int64_t>, dnnl::matmul> prims;
};

std::mutex mu;
std::map<ns_gpu*, std::unique_ptr<Dnn>> ctxs;

Dnn& dnn(ns_gpu* g) {
    auto& d = ctxs[g];
    if (!d) {
        d = std::make_unique<Dnn>();
        d->eng = dnnl::sycl_interop::make_engine(g->dev, g->ctx);
        d->st = dnnl::sycl_interop::make_stream(d->eng, g->q);
    }
    return *d;
}

memory usm(const memory::desc& md, const dnnl::engine& eng, const void* p) {
    return dnnl::sycl_interop::make_memory(md, eng, dnnl::sycl_interop::memory_kind::usm, const_cast<void*>(p));
}

// a 4-d float32 view: dims (B, H, rows, cols) with strides
memory::desc view(int64_t b, int64_t h, int64_t r, int64_t c, int64_t sb, int64_t sh, int64_t sr, int64_t sc) {
    return memory::desc({b, h, r, c}, memory::data_type::f32, memory::dims{sb, sh, sr, sc});
}

dnnl::matmul& prim(Dnn& d, const memory::desc& a, const memory::desc& w, const memory::desc& o, const std::vector<int64_t>& key) {
    auto it = d.prims.find(key);
    if (it == d.prims.end()) it = d.prims.emplace(key, dnnl::matmul(dnnl::matmul::primitive_desc(d.eng, a, w, o))).first;
    return it->second;
}
}  // namespace

extern "C" {

int ns_model3d_aam_silo_attention_supported(int64_t D, int64_t Dv, int causal) {
    return D > 0 && Dv > 0 && D <= 4096 && Dv <= 4096 ? 1 : 0;
}

int ns_model3d_aam_silo_attention(ns_gpu* g, const float* q, int64_t q_row, int64_t q_head, const float* k, int64_t k_row, int64_t k_head,
                                  const float* v, int64_t v_row, int64_t v_head, int64_t B, int64_t Sq, int64_t Skv, int64_t H, int64_t D,
                                  int64_t Dv, float scale, int causal, float* scratch, int64_t scratch_n, float* out, int64_t out_row) {
    NS_TRY
    if (B <= 0 || Sq <= 0 || Skv <= 0 || H <= 0) return 0;
    if (!ns_model3d_aam_silo_attention_supported(D, Dv, causal)) return ns_fail("ns_model3d_aam_silo_attention: shapes it does not take");
    if (causal && Sq != Skv) return ns_fail("ns_model3d_aam_silo_attention: a causal mask needs Sq = Skv");
    const int64_t per_row = B * H * Skv;
    if (scratch_n < per_row) return ns_fail("ns_model3d_aam_silo_attention: the scratch holds less than one query row of every head");
    const int64_t rows_max = std::min<int64_t>(Sq, scratch_n / per_row);
    std::lock_guard<std::mutex> l(mu);
    Dnn& d = dnn(g);
    for (int64_t r0 = 0; r0 < Sq; r0 += rows_max) {
        const int64_t rows = std::min(rows_max, Sq - r0);
        // scores [B, H, rows, Skv] = q . k^T
        const memory::desc qd = view(B, H, rows, D, Sq * q_row, q_head, q_row, 1);
        const memory::desc kd = view(B, H, D, Skv, Skv * k_row, k_head, 1, k_row);
        const memory::desc sd = view(B, H, rows, Skv, H * rows * Skv, rows * Skv, Skv, 1);
        prim(d, qd, kd, sd, {0, B, H, rows, Skv, D, q_row, q_head, k_row, k_head})
            .execute(d.st, {{DNNL_ARG_SRC, usm(qd, d.eng, q + r0 * q_row)}, {DNNL_ARG_WEIGHTS, usm(kd, d.eng, k)}, {DNNL_ARG_DST, usm(sd, d.eng, scratch)}});
        // softmax of each row (scale, and the causal mask: query r0 + r sees keys up to it)
        const int64_t R = B * H * rows;
        constexpr int WG = 256;
        g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) R * WG), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
            const int64_t row = it.get_group(0);
            const int l = (int) it.get_local_id(0);
            float* s = scratch + row * Skv;
            const int64_t n = causal ? (r0 + row % rows) + 1 : Skv;
            float m = -std::numeric_limits<float>::infinity();
            for (int64_t j = l; j < n; j += WG) m = sycl::fmax(m, s[j]);
            m = sycl::reduce_over_group(it.get_group(), m, sycl::maximum<float>());
            float sum = 0.0f;
            for (int64_t j = l; j < n; j += WG) {
                const float e = sycl::exp((s[j] - m) * scale);
                s[j] = e;
                sum += e;
            }
            sum = sycl::reduce_over_group(it.get_group(), sum, sycl::plus<float>());
            const float inv = 1.0f / sum;
            for (int64_t j = l; j < Skv; j += WG) s[j] = j < n ? s[j] * inv : 0.0f;
        });
        // out rows [B, H, rows, Dv] (written into [b Sq + r0 + i, h Dv + d]) = weights . v
        const memory::desc vd = view(B, H, Skv, Dv, Skv * v_row, v_head, v_row, 1);
        const memory::desc od = view(B, H, rows, Dv, Sq * out_row, Dv, out_row, 1);
        prim(d, sd, vd, od, {1, B, H, rows, Skv, Dv, v_row, v_head, out_row, Sq})
            .execute(d.st, {{DNNL_ARG_SRC, usm(sd, d.eng, scratch)}, {DNNL_ARG_WEIGHTS, usm(vd, d.eng, v)}, {DNNL_ARG_DST, usm(od, d.eng, out + r0 * out_row)}});
    }
    return 0;
    NS_CATCH
}

}  // extern "C"
