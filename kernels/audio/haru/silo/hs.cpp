// hs.cpp: Haru-4B's silo (hs.h). Started as a copy of haru.cpp's ns_audio_haru_gdn_rule; the change: each item keeps
// its column of the head's state in registers for all the call's rows (the shared kernel reads and writes it in
// device memory twice a row, so a prompt's pass of 128 rows waits on memory ~33,000 times in a row). The arithmetic is
// the same operations in the same order (the compiler may still fuse them differently: checked by the start stage,
// docs/ports/haru-4b.md).
#include "ns.h"
#include "ns_internal.hpp"
#include "hs.h"

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <cmath>

namespace {
constexpr int SG = 16;
constexpr int D = 128;

inline float sigmoid(float x) {
    return 1.0f / (1.0f + sycl::exp(-x));
}

// torch's softplus (beta 1, threshold 20)
inline float softplus(float x) {
    return x > 20.0f ? x : sycl::log1p(sycl::exp(x));
}

struct GdnRule;
}  // namespace

extern "C" int ns_haru_silo_gdn_supported(int64_t Hk, int64_t Hv, int64_t Dk, int64_t Dv) {
    return Dk == D && Dv == D && Hk > 0 && Hv % Hk == 0 ? 1 : 0;
}

extern "C" int ns_haru_silo_gdn_rule(ns_gpu* g, const float* qkv, int64_t ldq, const float* ba, int64_t ldb, const float* A_log, const float* dt_bias,
                                     float* state, int64_t R, int64_t Hk, int64_t Hv, int64_t Dk, int64_t Dv, float* out) {
    NS_TRY
    if (!ns_haru_silo_gdn_supported(Hk, Hv, Dk, Dv)) return ns_fail("ns_haru_silo_gdn_rule: heads of 128 keys and values, value heads a multiple of the key heads");
    if (R <= 0) return R == 0 ? 0 : ns_fail("ns_haru_silo_gdn_rule: bad rows");
    const int64_t rep = Hv / Hk, kd = Hk * Dk;
    const float scale = 1.0f / sycl::sqrt((float) Dk);
    namespace X = sycl::ext::oneapi::experimental;
    g->q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> qn(sycl::range<1>(D), h), kn(sycl::range<1>(D), h);
        h.parallel_for<GdnRule>(sycl::nd_range<1>(sycl::range<1>((size_t) Hv * D), sycl::range<1>(D)), X::properties{sycl::ext::intel::experimental::grf_size<256>},
                                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const int64_t hv = it.get_group(0), j = it.get_local_id(0), hk = hv / rep;
            float* S = state + hv * D * D;
            const float al = sycl::exp(A_log[hv]), db = dt_bias[hv];
            float s[D];
#pragma unroll
            for (int k = 0; k < D; k++) s[k] = S[k * D + j];
            for (int64_t r = 0; r < R; r++) {
                const float* row = qkv + r * ldq;
                const float qv = row[hk * D + j], kv = row[kd + hk * D + j], vv = row[2 * kd + hv * D + j];
                const float sq = sycl::reduce_over_group(it.get_group(), qv * qv, sycl::plus<float>());
                const float sk = sycl::reduce_over_group(it.get_group(), kv * kv, sycl::plus<float>());
                qn[j] = qv * sycl::rsqrt(sq + 1e-6f) * scale;
                kn[j] = kv * sycl::rsqrt(sk + 1e-6f);
                sycl::group_barrier(it.get_group());
                const float beta = sigmoid(ba[r * ldb + hv]);
                const float decay = sycl::exp(-al * softplus(ba[r * ldb + Hv + hv] + db));
                float mem = 0.0f;
#pragma unroll
                for (int k = 0; k < D; k++) mem += s[k] * kn[k];
                const float delta = (vv - decay * mem) * beta;
                float o = 0.0f;
#pragma unroll
                for (int k = 0; k < D; k++) {
                    s[k] = decay * s[k] + kn[k] * delta;
                    o += s[k] * qn[k];
                }
                out[r * Hv * D + hv * D + j] = o;
                sycl::group_barrier(it.get_group());
            }
#pragma unroll
            for (int k = 0; k < D; k++) S[k * D + j] = s[k];
        });
    });
    return 0;
    NS_CATCH
}

// A wide pass's attention: a sub-group a query row (all G query heads of its key head), a work-group eight rows of one
// key head (their key and value rows read together, through the cache), every key up to the row's position in one
// online softmax - no pieces, no scratch, no merge pass: the output gate applied at the end. Started from haru.cpp's
// ns_audio_haru_attn (the same per-key arithmetic: q . k in a lane's 16 values then the sub-group's sum, the running
// max and sum), whose pieces of keys a work-group each cost a prompt's pass of 128 rows ~3 ms a layer.
namespace {
constexpr int AD = 256;
constexpr int GMAX = 4;
constexpr int RB = 8;
struct AttnWide;
}  // namespace

extern "C" int ns_haru_silo_attn_supported(int64_t Hq, int64_t Hkv, int64_t Dh) {
    return Dh == AD && Hkv > 0 && Hq % Hkv == 0 && Hq / Hkv <= GMAX ? 1 : 0;
}

extern "C" int ns_haru_silo_attn(ns_gpu* g, const float* q, const void* kc, const void* vc, int64_t R, int64_t Hq, int64_t Hkv, int64_t Dh, int64_t T,
                                 int64_t p0, const float* gate, int64_t ldg, int64_t hs, float* out) {
    NS_TRY
    if (!ns_haru_silo_attn_supported(Hq, Hkv, Dh)) return ns_fail("ns_haru_silo_attn: heads of 256, at most 4 query heads a key head");
    if (R <= 0) return R == 0 ? 0 : ns_fail("ns_haru_silo_attn: bad rows");
    if (p0 + R > T) return ns_fail("ns_haru_silo_attn: past the cache's end");
    const int64_t G = Hq / Hkv, blocks = (R + RB - 1) / RB;
    const sycl::half* K = (const sycl::half*) kc;
    const sycl::half* V = (const sycl::half*) vc;
    const float scale = 1.0f / sycl::sqrt((float) AD);
    namespace X = sycl::ext::oneapi::experimental;
    g->q.parallel_for<AttnWide>(sycl::nd_range<1>(sycl::range<1>((size_t) (blocks * Hkv * RB * SG)), sycl::range<1>(RB * SG)),
                                X::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        const int64_t grp = it.get_group(0), hk = grp % Hkv, blk = grp / Hkv;
        auto sg = it.get_sub_group();
        const int sgi = (int) sg.get_group_linear_id(), lane = (int) sg.get_local_linear_id();
        const int64_t r = blk * RB + sgi;
        if (r >= R) return;
        const int64_t nk = p0 + r + 1;
        float qr[GMAX][16], o[GMAX][16], m[GMAX], l[GMAX];
#pragma unroll
        for (int gq = 0; gq < GMAX; gq++) {
            m[gq] = -INFINITY;
            l[gq] = 0.0f;
#pragma unroll
            for (int i = 0; i < 16; i++) {
                o[gq][i] = 0.0f;
                qr[gq][i] = gq < G ? q[r * Hq * AD + (hk * G + gq) * AD + lane * 16 + i] * scale : 0.0f;
            }
        }
        for (int64_t t = 0; t < nk; t++) {
            const sycl::half* kp = K + (hk * T + t) * AD + lane * 16;
            const sycl::half* vp = V + (hk * T + t) * AD + lane * 16;
            float kf[16], vf[16];
#pragma unroll
            for (int i = 0; i < 16; i++) {
                kf[i] = (float) kp[i];
                vf[i] = (float) vp[i];
            }
#pragma unroll
            for (int gq = 0; gq < GMAX; gq++) {
                if (gq >= G) break;
                float s = 0.0f;
#pragma unroll
                for (int i = 0; i < 16; i++) s += qr[gq][i] * kf[i];
                s = sycl::reduce_over_group(sg, s, sycl::plus<float>());
                const float mn = sycl::fmax(m[gq], s);
                const float a = sycl::exp(m[gq] - mn), e = sycl::exp(s - mn);
                l[gq] = l[gq] * a + e;
#pragma unroll
                for (int i = 0; i < 16; i++) o[gq][i] = o[gq][i] * a + e * vf[i];
                m[gq] = mn;
            }
        }
#pragma unroll
        for (int gq = 0; gq < GMAX; gq++) {
            if (gq >= G) break;
            const int64_t hh = hk * G + gq;
            const float inv = l[gq] > 0.0f ? 1.0f / l[gq] : 0.0f;
#pragma unroll
            for (int i = 0; i < 16; i++) {
                const int64_t d = lane * 16 + i;
                const float gt = gate[r * ldg + hh * hs + d];
                out[r * Hq * AD + hh * AD + d] = o[gq][i] * inv * (1.0f / (1.0f + sycl::exp(-gt)));
            }
        }
    });
    return 0;
    NS_CATCH
}
