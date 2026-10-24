// ace.cpp: the ACE-Step 1.5 engine's own kernels - what the shared diffusion kernels (nsd) do not have: the DiT's and
// its encoders' sliding-window attention (bidirectional: a row reads the keys within 128 of it on either side), and
// the Oobleck VAE's snake, whose divisor is a parameter of its own (beta) rather than the frequency (alpha).
// After nextcuda's ace.cu.
#include "ns.h"
#include "ns_internal.hpp"
#include "ace.h"

#include <sycl/sycl.hpp>
#include <cmath>

namespace {

constexpr int WARP = 32;
constexpr int HD = 128;          // the head size of every ACE-Step attention
constexpr int ROWS = 4;          // query rows a work-group (a sub-group each)

inline float bf(uint16_t b) { return sycl::bit_cast<float>((uint32_t) b << 16); }

inline uint16_t to_bf(float f) {
    // round to nearest even, NaN kept a NaN
    uint32_t u = sycl::bit_cast<uint32_t>(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t) ((u >> 16) | 0x40);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}

// a sub-group a query row, each lane 4 of the head's 128 features; online softmax over the window's keys - or, seg > 0,
// over the keys of the row's own segment (rows [seg i/seg, seg i/seg + seg): sequences of seg rows side by side)
void attn(sycl::queue& qu, const uint16_t* q, int64_t qs, const uint16_t* k, const uint16_t* v, int64_t kvs, int64_t S, int64_t Hq, int64_t Hkv,
          int64_t w, int64_t seg, float scale, uint16_t* out, int64_t os) {
    const int64_t nblk = (S + ROWS - 1) / ROWS;
    qu.parallel_for(sycl::nd_range<1>((size_t) (nblk * Hq) * (ROWS * WARP), ROWS * WARP), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
        auto sg = it.get_sub_group();
        const int lane = (int) sg.get_local_linear_id(), warp = (int) sg.get_group_linear_id();
        const int64_t grp = it.get_group(0);
        const int64_t i = (grp % nblk) * ROWS + warp, h = grp / nblk;
        if (i >= S) return;   // the whole sub-group: it holds one row
        const int64_t hk = h / (Hq / Hkv);
        const uint16_t* qp = q + i * qs + h * HD + lane * 4;
        float qv[4];
        for (int d = 0; d < 4; ++d) qv[d] = bf(qp[d]) * scale;
        int64_t j0 = w < 0 ? 0 : (i - w > 0 ? i - w : 0);
        int64_t j1 = w < 0 ? S - 1 : (i + w < S - 1 ? i + w : S - 1);
        if (seg > 0) {
            j0 = i / seg * seg;
            j1 = j0 + seg - 1 < S - 1 ? j0 + seg - 1 : S - 1;
        }
        float m = -INFINITY, l = 0.0f, o[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int64_t j = j0; j <= j1; ++j) {
            const uint16_t* kp = k + j * kvs + hk * HD + lane * 4;
            float s = qv[0] * bf(kp[0]) + qv[1] * bf(kp[1]) + qv[2] * bf(kp[2]) + qv[3] * bf(kp[3]);
            s = sycl::reduce_over_group(sg, s, sycl::plus<float>());
            const float mn = sycl::fmax(m, s);
            const float corr = sycl::exp(m - mn), p = sycl::exp(s - mn);
            const uint16_t* vp = v + j * kvs + hk * HD + lane * 4;
            l = l * corr + p;
            for (int d = 0; d < 4; ++d) o[d] = o[d] * corr + p * bf(vp[d]);
            m = mn;
        }
        const float inv = l > 0.0f ? 1.0f / l : 0.0f;
        uint16_t* op = out + i * os + h * HD + lane * 4;
        for (int d = 0; d < 4; ++d) op[d] = to_bf(o[d] * inv);
    });
}
}  // namespace

extern "C" {

int ns_audio_ace_attn_window(ns_gpu* g, const void* q, int64_t qs, const void* k, const void* v, int64_t kvs, int64_t S, int64_t Hq, int64_t Hkv,
                             int64_t w, float scale, void* out, int64_t os) {
    NS_TRY
    if (S <= 0) return 0;
    if (Hkv <= 0 || Hq % Hkv) return ns_fail("ns_audio_ace_attn_window: query heads are not a multiple of the key heads");
    if (qs % 4 || kvs % 4 || os % 4) return ns_fail("ns_audio_ace_attn_window: row strides must be multiples of 4 values");
    attn(g->q, (const uint16_t*) q, qs, (const uint16_t*) k, (const uint16_t*) v, kvs, S, Hq, Hkv, w, 0, scale, (uint16_t*) out, os);
    return 0;
    NS_CATCH
}

int ns_audio_ace_attn_segments(ns_gpu* g, const void* q, int64_t qs, const void* k, const void* v, int64_t kvs, int64_t S, int64_t Hq, int64_t Hkv,
                               int64_t seg, float scale, void* out, int64_t os) {
    NS_TRY
    if (S <= 0) return 0;
    if (seg <= 0) return ns_fail("ns_audio_ace_attn_segments: a segment of at least one row");
    if (Hkv <= 0 || Hq % Hkv) return ns_fail("ns_audio_ace_attn_segments: query heads are not a multiple of the key heads");
    if (qs % 4 || kvs % 4 || os % 4) return ns_fail("ns_audio_ace_attn_segments: row strides must be multiples of 4 values");
    attn(g->q, (const uint16_t*) q, qs, (const uint16_t*) k, (const uint16_t*) v, kvs, S, Hq, Hkv, -1, seg, scale, (uint16_t*) out, os);
    return 0;
    NS_CATCH
}

int ns_audio_ace_snake(ns_gpu* g, const float* x, int64_t C, int64_t L, const float* alpha, const float* inv_beta, float* out) {
    NS_TRY
    const int64_t n = C * L;
    if (n <= 0) return 0;
    g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0];
        const int64_t c = i / L;
        const float v = x[i], s = sycl::sin(alpha[c] * v);
        out[i] = v + inv_beta[c] * s * s;
    });
    return 0;
    NS_CATCH
}

}  // extern "C"
