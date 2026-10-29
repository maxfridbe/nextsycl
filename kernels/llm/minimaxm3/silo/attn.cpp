// attn.cpp: the minimaxm3 silo's prompt-pass attention. The shared kernel (../mm3.cpp, ns_llm_minimaxm3_attend) gives every
// (query, head) its own work-group and every thread one key's whole 128-value dot product, read with a stride of a key row: each
// key and value row is fetched once per head (16 times a group) and by lanes that are 256 bytes apart. This one takes a work-group a
// (query, key/value group) for the group's 16 heads at once: a tile of 64 keys is staged in local memory (coalesced loads) and
// used by all 16 heads, scores and the online softmax are per head, and the values are summed in key order as the shared
// kernel does. The scores' sums run in another order than the shared kernel's (a different rounding, ~1e-7 relative).
#include "ns.h"
#include "ns_internal.hpp"
#include "mm3s.h"

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdint>

namespace {
constexpr int HD = 128;     // values a head
constexpr int PER = 16;     // query heads a key/value group
constexpr int TILE = 64;    // keys a step
constexpr int WG = 256;     // threads a work-group: 16 heads x 16 lanes
constexpr int KS = HD + 1;  // a staged row's stride (floats): odd, so rows do not share a bank
}  // namespace

extern "C" {

int ns_mm3s_attend_supported(int64_t H, int64_t G, int64_t head) { return head == HD && G > 0 && H == G * PER; }

int ns_mm3s_attend(ns_gpu* g, const float* q, const uint16_t* kc, const uint16_t* vc, const int32_t* sel, int64_t K, float* out, int64_t T, int64_t H, int64_t G,
                   int64_t pos0, float scale) {
    NS_TRY
    if (!ns_mm3s_attend_supported(H, G, HD)) return ns_fail("ns_mm3s_attend: 16 heads a group of 128 values");
    g->q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> qs(sycl::range<1>(PER * HD), h);
        sycl::local_accessor<float, 1> kt(sycl::range<1>(TILE * KS), h);
        sycl::local_accessor<float, 1> pt(sycl::range<1>(PER * TILE), h);
        sycl::local_accessor<int32_t, 1> jj(sycl::range<1>(TILE), h);
        h.parallel_for(sycl::nd_range<1>((size_t) (T * G * WG), WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const int64_t grp = it.get_group(0);
            const int tid = (int) it.get_local_id(0);
            const int64_t t = grp / G, gi = grp % G;
            const int64_t pos = pos0 + t;
            const int hh = tid >> 4, l16 = tid & 15;  // this thread's head in the group and lane
            auto sg = it.get_sub_group();
            for (int e = tid; e < PER * HD; e += WG) qs[e] = q[(t * H + gi * PER + e / HD) * HD + e % HD];
            float m = -INFINITY, l = 0.f;
            float acc[HD / 16];
#pragma unroll
            for (int i = 0; i < HD / 16; ++i) acc[i] = 0.f;
            const int64_t total = sel ? K * 128 : pos + 1;
            for (int64_t c0 = 0; c0 < total; c0 += TILE) {
                const int n = (int) sycl::min<int64_t>(TILE, total - c0);
                if (tid < TILE) {
                    int64_t j = -1;
                    if (tid < n) {
                        const int64_t idx = c0 + tid;
                        j = idx;
                        if (sel) {
                            const int32_t b = sel[(t * G + gi) * K + idx / 128];
                            j = b < 0 ? -1 : (int64_t) b * 128 + idx % 128;
                        }
                        if (j > pos) j = -1;
                    }
                    jj[tid] = (int32_t) j;
                }
                sycl::group_barrier(it.get_group());
                // the keys' tile
                for (int e = tid; e < TILE * HD; e += WG) {
                    const int i = e / HD, d = e % HD;
                    const int32_t j = jj[i];
                    kt[i * KS + d] = j >= 0 ? (float) ((const sycl::half*) kc)[((int64_t) j * G + gi) * HD + d] : 0.f;
                }
                sycl::group_barrier(it.get_group());
                // scores of this head for keys l16, l16 + 16, ...
                float cm = -INFINITY;
#pragma unroll
                for (int u = 0; u < TILE / 16; ++u) {
                    const int k = l16 + 16 * u;
                    float s = -INFINITY;
                    if (jj[k] >= 0) {
                        float a = 0.f;
                        for (int d = 0; d < HD; ++d) a += qs[hh * HD + d] * kt[k * KS + d];
                        s = a * scale;
                    }
                    pt[hh * TILE + k] = s;
                    cm = sycl::fmax(cm, s);
                }
                // the head's 16 lanes sit in one half of a sub-group: xor masks below 16 stay inside it
#pragma unroll
                for (int mk = 8; mk >= 1; mk >>= 1) cm = sycl::fmax(cm, sycl::permute_group_by_xor(sg, cm, mk));
                const float nm = sycl::fmax(m, cm);
                const float corr = m == -INFINITY ? 0.f : sycl::exp(m - nm);
                float ls = 0.f;
#pragma unroll
                for (int u = 0; u < TILE / 16; ++u) {
                    const int k = l16 + 16 * u;
                    const float s = pt[hh * TILE + k];
                    const float ev = (nm > -INFINITY && s != -INFINITY) ? sycl::exp(s - nm) : 0.f;
                    pt[hh * TILE + k] = ev;
                    ls += ev;
                }
#pragma unroll
                for (int mk = 8; mk >= 1; mk >>= 1) ls += sycl::permute_group_by_xor(sg, ls, mk);
                l = l * corr + ls;
#pragma unroll
                for (int i = 0; i < HD / 16; ++i) acc[i] *= corr;
                m = nm;
                sycl::group_barrier(it.get_group());
                // the values' tile replaces the keys'
                for (int e = tid; e < TILE * HD; e += WG) {
                    const int i = e / HD, d = e % HD;
                    const int32_t j = jj[i];
                    kt[i * KS + d] = j >= 0 ? (float) ((const sycl::half*) vc)[((int64_t) j * G + gi) * HD + d] : 0.f;
                }
                sycl::group_barrier(it.get_group());
                // acc[i] holds value d = l16 + 16 i; the keys in order
                for (int k = 0; k < n; ++k) {
                    const float p = pt[hh * TILE + k];
#pragma unroll
                    for (int i = 0; i < HD / 16; ++i) acc[i] += p * kt[k * KS + l16 + 16 * i];
                }
                sycl::group_barrier(it.get_group());
            }
            float* o = out + (t * H + gi * PER + hh) * HD;
#pragma unroll
            for (int i = 0; i < HD / 16; ++i) o[l16 + 16 * i] = l > 0.f ? acc[i] / l : 0.f;
        });
    });
    return 0;
    NS_CATCH
}

}  // extern "C"
