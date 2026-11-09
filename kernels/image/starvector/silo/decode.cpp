// decode.cpp: the StarVector silo's decode attention and layer norm (silo.h). Attention: flash-decoding for StarCoder's multi-query attention: the
// N keys of a beam's cache are cut in chunks of 128; one work-group a (beam, head, chunk) scores its chunk (a work-item a
// key, the query in local memory), takes the chunk's max and sum of exponentials, and weights the chunk's values (a
// work-item a feature); a second pass combines each head's chunks with the usual max-rescaling. Half in and out,
// float32 sums, as the shared kernel does it.
#include "ns.h"
#include "ns_internal.hpp"
#include "silo.h"

#include <sycl/sycl.hpp>

#include <cmath>
#include <map>
#include <memory>
#include <mutex>

namespace {
using half = sycl::half;
constexpr int CH = 128;           // keys a chunk = work-items a group
constexpr int HD = 128;           // the head
constexpr int64_t MAXN = 8192, MAXB = 8, MAXH = 64;
constexpr int64_t MAXCHUNKS = MAXN / CH;

struct Ptrs {                      // the beams' caches, captured by value
    const half* k[MAXB];
    const half* v[MAXB];
};

struct State {
    float* part = nullptr;         // [beam, head, chunk, 2 + HD]: the chunk's max, sum, weighted values
};
std::mutex mu;
std::map<ns_gpu*, std::unique_ptr<State>> states;

State& state(ns_gpu* g) {
    auto& s = states[g];
    if (!s) {
        s = std::make_unique<State>();
        s->part = sycl::malloc_device<float>((size_t) (MAXB * MAXH * MAXCHUNKS * (2 + HD)), g->q);
        if (!s->part) throw std::runtime_error("starvector attention: no memory for its scratch");
    }
    return *s;
}
}  // namespace

extern "C" {

int ns_image_starvector_silo_attn_supported(ns_gpu*, int64_t nb, int64_t N, int64_t H, int64_t D) {
    return D == HD && nb >= 1 && nb <= MAXB && N >= 1 && N <= MAXN && H >= 1 && H <= MAXH ? 1 : 0;
}

int ns_image_starvector_silo_decode_attn(ns_gpu* g, const void* q, const void* const* k, const void* const* v, int64_t nb, int64_t N, int64_t H, int64_t D, void* out) {
    NS_TRY
    if (!ns_image_starvector_silo_attn_supported(g, nb, N, H, D)) return ns_fail("ns_image_starvector_silo_decode_attn: D 128, up to 8 beams, 8192 keys, 64 heads");
    std::lock_guard<std::mutex> lock(mu);
    float* part = state(g).part;
    const int64_t nc = (N + CH - 1) / CH;
    const float scale = 1.0f / sycl::sqrt((float) HD);
    Ptrs ps{};
    for (int64_t b = 0; b < nb; ++b) {
        ps.k[b] = static_cast<const half*>(k[b]);
        ps.v[b] = static_cast<const half*>(v[b]);
    }
    const half* qh = static_cast<const half*>(q);
    half* oh = static_cast<half*>(out);
    g->q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> ql(sycl::range<1>(HD), h);
        sycl::local_accessor<float, 1> pl(sycl::range<1>(CH), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (nb * H * nc * CH)), sycl::range<1>(CH)), [=](sycl::nd_item<1> it) {
            const int64_t gi = it.get_group(0), c = gi % nc, hh = (gi / nc) % H, b = gi / (nc * H);
            const int t = (int) it.get_local_id(0);
            ql[t] = (float) qh[(b * H + hh) * HD + t];
            it.barrier(sycl::access::fence_space::local_space);
            const int64_t j = c * CH + t;
            const bool valid = j < N;
            float s = -INFINITY;
            if (valid) {
                const sycl::vec<half, 8>* kr = reinterpret_cast<const sycl::vec<half, 8>*>(ps.k[b] + j * HD);
                float acc = 0.0f;
                for (int d8 = 0; d8 < HD / 8; ++d8) {
                    const sycl::vec<half, 8> kv = kr[d8];
                    for (int e = 0; e < 8; ++e) acc += ql[d8 * 8 + e] * (float) kv[e];
                }
                s = acc * scale;
            }
            const float m = sycl::reduce_over_group(it.get_group(), s, sycl::maximum<float>());
            const float p = valid ? sycl::exp(s - m) : 0.0f;
            const float l = sycl::reduce_over_group(it.get_group(), p, sycl::plus<float>());
            pl[t] = p;
            it.barrier(sycl::access::fence_space::local_space);
            const int64_t n = sycl::min<int64_t>(CH, N - c * CH);
            const half* vr = ps.v[b] + c * CH * HD + t;
            float acc = 0.0f;
            for (int u = 0; u < n; ++u) acc += pl[u] * (float) vr[(int64_t) u * HD];
            float* o = part + ((b * H + hh) * nc + c) * (2 + HD);
            if (t == 0) { o[0] = m; o[1] = l; }
            o[2 + t] = acc;
        });
    });
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (nb * H * HD)), sycl::range<1>(HD)), [=](sycl::nd_item<1> it) {
        const int64_t bh = it.get_group(0);
        const int t = (int) it.get_local_id(0);
        const float* o = part + bh * nc * (2 + HD);
        float m = -INFINITY;
        for (int64_t c = 0; c < nc; ++c) m = sycl::fmax(m, o[c * (2 + HD)]);
        float l = 0.0f, acc = 0.0f;
        for (int64_t c = 0; c < nc; ++c) {
            const float w = sycl::exp(o[c * (2 + HD)] - m);
            l += o[c * (2 + HD) + 1] * w;
            acc += o[c * (2 + HD) + 2 + t] * w;
        }
        oh[bh * HD + t] = (half) (acc / l);
    });
    return 0;
    NS_CATCH
}

int ns_image_starvector_silo_ln_supported(ns_gpu*, int64_t M, int64_t C) {
    return M >= 1 && C >= 256 && C <= 4096 && C % 256 == 0 ? 1 : 0;
}

// LayerNorm, a work-group of 256 a row: each work-item holds its C / 256 values (at most 16) in registers, the mean and the
// variance (the two-pass form, as the shared kernel's) are group reductions.
int ns_image_starvector_silo_layer_norm(ns_gpu* g, const float* x, int64_t M, int64_t C, const float* weight, const float* bias, float eps, void* out) {
    NS_TRY
    if (!ns_image_starvector_silo_ln_supported(g, M, C)) return ns_fail("ns_image_starvector_silo_layer_norm: C a multiple of 256, up to 4096");
    constexpr int WG = 256;
    const int per = (int) (C / WG);
    half* oh = static_cast<half*>(out);
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (M * WG)), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
        const int64_t r = it.get_group(0);
        const int t = (int) it.get_local_id(0);
        const float* xr = x + r * C;
        float v[16];
        float s = 0.0f;
        for (int e = 0; e < per; ++e) {
            v[e] = xr[e * WG + t];
            s += v[e];
        }
        const float mean = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>()) / (float) C;
        float q = 0.0f;
        for (int e = 0; e < per; ++e) {
            const float d = v[e] - mean;
            q += d * d;
        }
        const float inv = sycl::rsqrt(sycl::reduce_over_group(it.get_group(), q, sycl::plus<float>()) / (float) C + eps);
        for (int e = 0; e < per; ++e) {
            const int i = e * WG + t;
            float y = (v[e] - mean) * inv;
            if (weight) y *= weight[i];
            if (bias) y += bias[i];
            oh[r * C + i] = (half) y;
        }
    });
    return 0;
    NS_CATCH
}

}  // extern "C"
