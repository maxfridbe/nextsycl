// attn.cpp: the MOSS-Image silo's causal attention for the text encoder (silo.h). Query blocks of 512 rows, each against
// the keys up to its end: scores = Q K^T (half, float32 sums, oneDNN, the 4 key/value heads broadcast over their 4 query
// heads), a float32 softmax with the causal mask (scale folded in; masked entries 0), then P V. The shapes depend only
// on the block, not on the prompt (the rows are padded to whole blocks, the padded keys and values zero), so a
// primitive made for one prompt serves the next.
#include "ns.h"
#include "ns_internal.hpp"
#include "silo.h"

#include <sycl/sycl.hpp>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>

#include <cmath>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

namespace {
using dnnl::memory;
using dt = memory::data_type;
using half = sycl::half;
constexpr int64_t C = 512;          // query rows a block
constexpr int64_t MAXT = 4096;
constexpr int WG = 256;

struct State {
    dnnl::engine eng;
    dnnl::stream st;
    std::map<std::vector<int64_t>, dnnl::matmul> prims;
    // padded to whole blocks: q, k, v in half; scores float32 [hq, C, MAXT]; probabilities half; the output float32
    half *qh = nullptr, *kh = nullptr, *vh = nullptr, *ph = nullptr;
    float *sc = nullptr, *ob = nullptr;
    int64_t hq = 0, hkv = 0, d = 0;
};

std::mutex mu;
std::map<ns_gpu*, std::unique_ptr<State>> states;

memory usm(const memory::desc& md, const dnnl::engine& e, const void* p) {
    return dnnl::sycl_interop::make_memory(md, e, dnnl::sycl_interop::memory_kind::usm, const_cast<void*>(p));
}

State& state(ns_gpu* g, int64_t hq, int64_t hkv, int64_t d) {
    auto& s = states[g];
    if (s && (s->hq != hq || s->hkv != hkv || s->d != d)) {
        sycl::free(s->qh, g->q); sycl::free(s->kh, g->q); sycl::free(s->vh, g->q);
        sycl::free(s->ph, g->q); sycl::free(s->sc, g->q); sycl::free(s->ob, g->q);
        s.reset();
    }
    if (!s) {
        s = std::make_unique<State>();
        s->eng = dnnl::sycl_interop::make_engine(g->dev, g->ctx);
        s->st = dnnl::sycl_interop::make_stream(s->eng, g->q);
        s->hq = hq; s->hkv = hkv; s->d = d;
        s->qh = sycl::malloc_device<half>((size_t) (MAXT * hq * d), g->q);
        s->kh = sycl::malloc_device<half>((size_t) (MAXT * hkv * d), g->q);
        s->vh = sycl::malloc_device<half>((size_t) (MAXT * hkv * d), g->q);
        s->ph = sycl::malloc_device<half>((size_t) (hq * C * MAXT), g->q);
        s->sc = sycl::malloc_device<float>((size_t) (hq * C * MAXT), g->q);
        s->ob = sycl::malloc_device<float>((size_t) (MAXT * hq * d), g->q);
        if (!s->qh || !s->kh || !s->vh || !s->ph || !s->sc || !s->ob) throw std::runtime_error("moss attention: no memory for its scratch");
    }
    return *s;
}

dnnl::matmul& prim(State& s, bool second, int64_t skv, const memory::desc& a, const memory::desc& b, const memory::desc& c) {
    const std::vector<int64_t> key{second ? 1 : 0, skv, s.hq, s.hkv, s.d};
    auto it = s.prims.find(key);
    if (it == s.prims.end()) it = s.prims.emplace(key, dnnl::matmul(dnnl::matmul::primitive_desc(s.eng, a, b, c))).first;
    return it->second;
}
}  // namespace

extern "C" {

int ns_image_moss_silo_attn_supported(ns_gpu*, int64_t t, int64_t hq, int64_t hkv, int64_t d) {
    return t >= 1 && t <= MAXT && d == 256 && hkv > 0 && hq % hkv == 0 ? 1 : 0;
}

int ns_image_moss_silo_attn_causal(ns_gpu* g, const float* q, const float* k, const float* v, int64_t t, int64_t hq, int64_t hkv, int64_t d, int64_t qs, int64_t kvs,
                                   float* out) {
    NS_TRY
    if (!ns_image_moss_silo_attn_supported(g, t, hq, hkv, d)) return ns_fail("ns_image_moss_silo_attn_causal: t up to 4096, d 256, hq a multiple of hkv");
    std::lock_guard<std::mutex> lock(mu);
    State& s = state(g, hq, hkv, d);
    const int64_t nb = (t + C - 1) / C, np = nb * C, G = hq / hkv;
    const float scale = 1.0f / sycl::sqrt((float) d);
    half *qh = s.qh, *kh = s.kh, *vh = s.vh, *ph = s.ph;
    float *sc = s.sc, *ob = s.ob;
    // half copies; the padded rows zero (a padded key's value must not be a stale NaN times a zero probability)
    g->q.parallel_for(sycl::range<1>((size_t) (np * hq * d)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0], r = i / (hq * d), c = i % (hq * d);
        qh[i] = r < t ? (half) q[r * qs + c] : (half) 0.0f;
    });
    g->q.parallel_for(sycl::range<1>((size_t) (np * hkv * d)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0], r = i / (hkv * d), c = i % (hkv * d);
        kh[i] = r < t ? (half) k[r * kvs + c] : (half) 0.0f;
        vh[i] = r < t ? (half) v[r * kvs + c] : (half) 0.0f;
    });
    for (int64_t b = 0; b < nb; ++b) {
        const int64_t skv = (b + 1) * C;
        // scores [hkv, G, C, skv] = q block {hkv, G, C, d} . K^T {hkv, 1, d, skv} (K's rows hkv x d apart)
        memory::desc qd({hkv, G, C, d}, dt::f16, memory::dims{G * d, d, hq * d, 1});
        memory::desc kd({hkv, 1, d, skv}, dt::f16, memory::dims{d, d, 1, hkv * d});
        memory::desc sd({hkv, G, C, skv}, dt::f32, memory::dims{G * C * skv, C * skv, skv, 1});
        auto& m1 = prim(s, false, skv, qd, kd, sd);
        m1.execute(s.st, {{DNNL_ARG_SRC, usm(qd, s.eng, qh + b * C * hq * d)}, {DNNL_ARG_WEIGHTS, usm(kd, s.eng, kh)}, {DNNL_ARG_DST, usm(sd, s.eng, sc)}});
        // softmax over a row's keys up to the query's own position: a work-group a row; masked entries 0
        g->q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (hq * C * WG)), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
                const int64_t row = it.get_group(0), r = row % C, qi = b * C + r;
                const int l = (int) it.get_local_id(0);
                const float* x = sc + row * skv;
                half* p = ph + row * skv;
                float m = -INFINITY;
                for (int64_t j = l; j <= qi; j += WG) m = sycl::fmax(m, x[j] * scale);
                m = sycl::reduce_over_group(it.get_group(), m, sycl::maximum<float>());
                float sum = 0.0f;
                for (int64_t j = l; j <= qi; j += WG) sum += sycl::exp(x[j] * scale - m);
                sum = sycl::reduce_over_group(it.get_group(), sum, sycl::plus<float>());
                const float inv = 1.0f / sum;
                for (int64_t j = l; j < skv; j += WG) p[j] = j <= qi ? (half) (sycl::exp(x[j] * scale - m) * inv) : (half) 0.0f;
            });
        });
        // out block {hkv, G, C, d} = P {hkv, G, C, skv} . V {hkv, 1, skv, d}
        memory::desc pd({hkv, G, C, skv}, dt::f16, memory::dims{G * C * skv, C * skv, skv, 1});
        memory::desc vd({hkv, 1, skv, d}, dt::f16, memory::dims{d, d, hkv * d, 1});
        memory::desc od({hkv, G, C, d}, dt::f32, memory::dims{G * d, d, hq * d, 1});
        auto& m2 = prim(s, true, skv, pd, vd, od);
        m2.execute(s.st, {{DNNL_ARG_SRC, usm(pd, s.eng, ph)}, {DNNL_ARG_WEIGHTS, usm(vd, s.eng, vh)}, {DNNL_ARG_DST, usm(od, s.eng, ob + b * C * hq * d)}});
    }
    g->q.memcpy(out, ob, (size_t) (t * hq * d) * sizeof(float));
    return 0;
    NS_CATCH
}

}  // extern "C"
