// attn.cpp: the inkling2 silo's decode attention (silo.h: ns_ink2s_attn_part) - the base kernel's split form (ink.cpp)
// with shorter splits (64 keys at absolute positions: more work-groups at a short context), without its merge (the
// merge, the sink, the output gate and the quantization are ns_ink2s_merge_gate_xq). A work-group a (row, key head,
// split): a sub-group a query head of the three; the split's keys a tile in local memory; the logits a lane a key (no
// reduction a key), the softmax of the split's 64 at once, then the values with the lanes over the dims.
#include "ns.h"
#include "ns_internal.hpp"
#include "silo.h"

#include <sycl/sycl.hpp>
#include <cmath>

namespace {
using half = sycl::half;
constexpr int SG = 16, D = 128, DL = D / SG, KB = 64, G = 3;

void range(int64_t T, int64_t p0, int64_t win, int64_t& s0, int64_t& ns) {
    const int64_t lo = win > 0 ? (p0 + 1 - win > 0 ? p0 + 1 - win : 0) : 0, hi = p0 + T;
    s0 = lo / KB;
    ns = (hi - 1) / KB - s0 + 1;
}
}  // namespace

extern "C" {

int64_t ns_ink2s_attn_splits(int64_t T, int64_t p0, int64_t win) {
    int64_t s0, ns;
    range(T, p0, win, s0, ns);
    return ns;
}

static int attn_part(ns_gpu* g, const float* qp, const void* kcv, const void* vcv, const float* r, const float* relb, int64_t E, int64_t R, float* part,
                     int64_t T, int64_t Hq, int64_t Hkv, int64_t cap, int64_t p0, int64_t win, float scale, bool block) {
    NS_TRY
    if (Hq != G * Hkv || R > SG || T < 1 || T > 16) return ns_fail("ns_ink2s_attn_part: 3 query heads a key head, a bias of at most 16, 1..16 rows");
    int64_t s0, ns;
    range(T, p0, win, s0, ns);
    const half* kc = (const half*) kcv;
    const half* vc = (const half*) vcv;
    constexpr int WG = G * SG;
    g->q.submit([&](sycl::handler& h) {
        sycl::local_accessor<half, 1> ks(sycl::range<1>((size_t) KB * D), h);
        sycl::local_accessor<half, 1> vs(sycl::range<1>((size_t) KB * D), h);
        // each sub-group's query (scaled), bias query, the split's probabilities
        sycl::local_accessor<float, 1> qs(sycl::range<1>((size_t) G * (D + SG + KB)), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (T * Hkv * ns) * WG), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            auto sg = it.get_sub_group();
            const int lane = (int) sg.get_local_linear_id(), sgi = (int) sg.get_group_linear_id();
            int64_t gi = it.get_group(0);
            const int64_t sp = gi % ns;
            gi /= ns;
            const int64_t hk = gi % Hkv, rr = gi / Hkv, hq = hk * G + sgi;
            // a block (the drafter's): every row sees the window from the block's start and the whole block
            const int64_t pos = p0 + rr, wpos = block ? p0 : pos;
            const int64_t first = win > 0 ? sycl::max<int64_t>(0, wpos + 1 - win) : 0, end = block ? p0 + T : pos + 1;
            const int64_t lo = sycl::max<int64_t>(first, (s0 + sp) * KB), hi = sycl::min<int64_t>(end, (s0 + sp + 1) * KB);
            const int64_t nk = hi > lo ? hi - lo : 0;
            float* qx = qs.get_multi_ptr<sycl::access::decorated::no>().get() + sgi * (D + SG + KB);
            float* rx = qx + D;
            float* px = rx + SG;
#pragma unroll
            for (int i = 0; i < DL; ++i) qx[lane * DL + i] = qp[(rr * Hq + hq) * D + lane * DL + i] * scale;
            rx[lane] = lane < R && r ? r[(rr * Hq + hq) * R + lane] : 0.0f;
            // the split's keys and values into local memory: a sub-group a key at a time, its lanes 8 dims each (16-byte
            // loads); the ring's slot once a key, the split's keys contiguous in it but for one wrap
            {
                const int64_t s0k = lo % cap;
                const int nki = (int) nk;
                for (int jj = sgi; jj < nki; jj += G) {
                    int64_t slot = s0k + jj;
                    if (slot >= cap) slot -= cap;
                    const sycl::uint4* ksrc = (const sycl::uint4*) (kc + (hk * cap + slot) * D) + lane;
                    const sycl::uint4* vsrc = (const sycl::uint4*) (vc + (hk * cap + slot) * D) + lane;
                    ((sycl::uint4*) (ks.get_multi_ptr<sycl::access::decorated::no>().get() + jj * D))[lane] = *ksrc;
                    ((sycl::uint4*) (vs.get_multi_ptr<sycl::access::decorated::no>().get() + jj * D))[lane] = *vsrc;
                }
            }
            sycl::group_barrier(it.get_group());
            const half* kp = ks.get_multi_ptr<sycl::access::decorated::no>().get();
            const half* vp = vs.get_multi_ptr<sycl::access::decorated::no>().get();
            // the logits: a lane a key (KB / SG each), its dims in 16 chunks of 8 taken from a lane-rotated start (the
            // lanes' rows on different banks), the bias over the R relative dims
            constexpr int PK = KB / SG;
            float lg[PK];
            float m = -INFINITY;
#pragma unroll
            for (int u = 0; u < PK; ++u) {
                const int jj = lane + u * SG;
                float v = -INFINITY;
                if (jj < nk) {
                    float dot = 0.0f;
#pragma unroll
                    for (int c = 0; c < SG; ++c) {
                        const int ci = (c + lane) & (SG - 1);
                        const sycl::vec<half, 8> kv8 = *(const sycl::vec<half, 8>*) (kp + jj * D + ci * DL);
#pragma unroll
                        for (int i = 0; i < DL; ++i) dot = sycl::fma(qx[ci * DL + i], (float) kv8[i], dot);
                    }
                    const int64_t d = pos - (lo + jj);
                    if (d >= 0 && d < E) {
                        for (int i = 0; i < R; ++i) dot = sycl::fma(rx[i], relb[d * R + i], dot);
                    }
                    v = dot;
                }
                lg[u] = v;
                m = sycl::fmax(m, v);
            }
            m = sycl::reduce_over_group(sg, m, sycl::maximum<float>());
            float s = 0.0f;
#pragma unroll
            for (int u = 0; u < PK; ++u) {
                const float p = nk > 0 && lane + u * SG < nk ? sycl::exp(lg[u] - m) : 0.0f;
                px[lane + u * SG] = p;
                s += p;
            }
            s = sycl::reduce_over_group(sg, s, sycl::plus<float>());
            sycl::group_barrier(sg);
            // the values weighted: lanes over the dims, the keys in order
            float acc[DL];
#pragma unroll
            for (int i = 0; i < DL; ++i) acc[i] = 0.0f;
            for (int64_t jj = 0; jj < nk; ++jj) {
                const float p = px[jj];
                const sycl::vec<half, 8> vv = *(const sycl::vec<half, 8>*) (vp + jj * D + lane * DL);
#pragma unroll
                for (int i = 0; i < DL; ++i) acc[i] = sycl::fma(p, (float) vv[i], acc[i]);
            }
            float* pp = part + ((rr * Hq + hq) * ns + sp) * (D + 2);
#pragma unroll
            for (int i = 0; i < DL; ++i) pp[lane * DL + i] = acc[i];
            if (lane == 0) {
                pp[D] = m;
                pp[D + 1] = s;
            }
        });
    });
    return 0;
    NS_CATCH
}

int ns_ink2s_attn_part(ns_gpu* g, const float* qp, const void* kcv, const void* vcv, const float* r, const float* relb, int64_t E, int64_t R, float* part,
                       int64_t T, int64_t Hq, int64_t Hkv, int64_t cap, int64_t p0, int64_t win, float scale) {
    return attn_part(g, qp, kcv, vcv, r, relb, E, R, part, T, Hq, Hkv, cap, p0, win, scale, false);
}

int ns_ink2s_attn_part_block(ns_gpu* g, const float* q, const void* kc, const void* vc, float* part, int64_t T, int64_t Hq, int64_t Hkv, int64_t cap, int64_t p0,
                             int64_t win, float scale) {
    return attn_part(g, q, kc, vc, nullptr, nullptr, 0, 0, part, T, Hq, Hkv, cap, p0, win, scale, true);
}

}  // extern "C"
