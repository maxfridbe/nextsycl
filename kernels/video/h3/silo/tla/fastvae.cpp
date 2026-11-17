// tla/fastvae.cpp: the H3 video engine's silo (silo.h) - what Open-H3-Regenerate-2K's fast video decoder (a 24-layer
// non-causal distillation of the H3 decoder: 5 x 16 x 16-token windows, shifted on odd layers, and 5 global tokens a
// 5-latent block; crates/h3/src/fastvae.rs) needs beyond the shared diffusion kernels:
//   varlen     every window's queries over its own tokens and the global tokens of the blocks it overlaps, as one
//              variable-length problem: ARK's bfloat16 prefill kernel on sycl-tla (kernels/diffusion/flash/flash.cpp's,
//              heads of 64) with cu_seqlens - upstream runs FlashAttention-4's varlen form the same way (no padding, no
//              mask; a window's rows are packed by the indexed row copy)
//   glob       a block's 5 global tokens attend to every token of their block (72,000 at 2560x1440): the decoder runs a
//              layer a temporal window at a time, so the block's keys come in parts; each part is a split-key pass (256
//              keys a work-group, the scores' max, sum and weighted values) folded into a running state - the softmax
//              over the whole block, exactly
//   rows       an indexed copy of 16-bit rows (16 bytes a work-item)
// sycl-tla launches on compat's default queue of the calling thread's device: use_queue points the thread at the card of
// the queue passed and makes that queue its default (flashvsr's silo does the same).
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <mutex>
#include <string>

#include <sycl/sycl.hpp>
#include "sycl_tla_sdpa.hpp"
#include "../silo.h"

namespace {
thread_local std::string g_err;
std::mutex g_mu;

constexpr int D = 64;          // head size
constexpr int ST = D + 2;      // a state row: max, sum, D values
constexpr int KCH = 256;       // keys a work-group (glob)
constexpr int MAXS = 8;        // query rows glob takes
constexpr float SCALE = 0.125f;  // 1 / sqrt(64)

int fail(const std::string& m) {
    g_err = m;
    return -1;
}

void use_queue(sycl::queue* q) {
    std::lock_guard<std::mutex> l(g_mu);
    const sycl::device dev = q->get_device();
    const unsigned n = compat::device_count();
    for (unsigned i = 0; i < n; ++i) {
        if (static_cast<const sycl::device&>(compat::get_device(i)) == dev) {
            compat::select_device(i);
            if (compat::get_default_queue() != *q) compat::set_default_queue(*q);
            return;
        }
    }
    throw std::runtime_error("the queue's GPU is not one sycl-tla sees");
}

inline float bf(uint16_t h) {
    return sycl::bit_cast<float>((uint32_t) h << 16);
}

inline uint16_t to_bf(float f) {
    uint32_t u = sycl::bit_cast<uint32_t>(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t) ((u >> 16) | 0x40);   // NaN stays NaN
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}
}  // namespace

extern "C" {

const char* ns_video_h3_silo_error(void) { return g_err.c_str(); }

int ns_video_h3_silo_supported(void* queue) try {
    auto* q = static_cast<sycl::queue*>(queue);
    return q && q->get_device().is_gpu() ? 1 : 0;
} catch (...) {
    return 0;
}

int ns_video_h3_silo_rows(void* queue, const void* src, int64_t sstride, const int32_t* sidx, void* dst, int64_t dstride,
                          const int32_t* didx, int64_t n, int64_t width) try {
    if (n <= 0) return 0;
    if (width <= 0 || width % 8 || sstride % 8 || dstride % 8) return fail("silo rows: width and strides in multiples of 8 elements");
    auto* qu = static_cast<sycl::queue*>(queue);
    using V = sycl::vec<uint32_t, 4>;
    const V* s = static_cast<const V*>(src);
    V* d = static_cast<V*>(dst);
    const int64_t w = width / 8, ss = sstride / 8, ds = dstride / 8;
    qu->parallel_for(sycl::range<2>((size_t) n, (size_t) w), [=](sycl::id<2> id) {
        const int64_t i = id[0], c = id[1];
        const int64_t a = sidx ? sidx[i] : i, b = didx ? didx[i] : i;
        d[b * ds + c] = s[a * ss + c];
    });
    return 0;
} catch (const std::exception& e) {
    return fail(std::string("silo rows: ") + e.what());
}

int ns_video_h3_silo_varlen(void* queue, const void* q, const void* k, const void* v, void* out, int64_t seqs, const int32_t* cu_q,
                            const int32_t* cu_k, const int32_t* cu_zero, int64_t max_q, int64_t max_k, int64_t total_q, int64_t total_k,
                            int64_t H) try {
    if (seqs <= 0 || H <= 0 || max_q <= 0 || max_k <= 0) return fail("silo varlen: an empty problem");
    if (total_q * H * D > INT32_MAX || total_k * H * D > INT32_MAX) return fail("silo varlen: over 2^31 elements (fewer rows a call)");
    use_queue(static_cast<sycl::queue*>(queue));
    ark::detail::Options o;
    o.q = q; o.k = k; o.v = v; o.o = out;
    o.batch = (int) seqs;
    o.num_heads_q = o.num_heads_kv = (int) H;
    o.seq_len_qo = (int) max_q;
    o.seq_len_kv = (int) max_k;
    o.head_size_qk = o.head_size_vo = D;
    o.softmax_scale = SCALE;
    o.is_causal = false;
    o.varlen = true;
    o.total_seqlen_q = (int) total_q;
    o.total_seqlen_kv = (int) total_k;
    o.max_seqlen_q = (int) max_q;
    o.max_seqlen_kv = (int) max_k;
    o.cu_seqlens_q = cu_q;
    o.cu_seqlens_k = cu_k;
    o.cu_seqlens_kv_cache = cu_zero;     // non-null: ARK would fill its own on the host otherwise
    o.use_tensor_strides = true;
    const int rs = (int) (H * D);
    o.q_stride_s = rs; o.q_stride_h = D; o.q_stride_b = 0;
    o.k_stride_s = rs; o.k_stride_h = D; o.k_stride_b = 0;
    o.v_stride_s = rs; o.v_stride_h = D; o.v_stride_b = 0;
    o.o_stride_s = rs; o.o_stride_h = D; o.o_stride_b = 0;
    using bf16 = cute::bfloat16_t;
    if (ark::detail::launch_prefill_kernel_64<false, false, true, bf16, bf16, bf16>(o) != 0) return fail("silo varlen: the kernel refused the problem");
    return 0;
} catch (const std::exception& e) {
    return fail(std::string("silo varlen: ") + e.what());
}

int ns_video_h3_silo_glob_init(void* queue, float* state, int64_t count) try {
    if (count <= 0) return 0;
    auto* qu = static_cast<sycl::queue*>(queue);
    qu->parallel_for(sycl::range<1>((size_t) (count * ST)), [=](sycl::id<1> i) {
        state[i] = (i % ST == 0) ? -INFINITY : 0.0f;
    });
    return 0;
} catch (const std::exception& e) {
    return fail(std::string("silo glob_init: ") + e.what());
}

int ns_video_h3_silo_glob(void* queue, const void* qg, int64_t qs, const void* k, const void* v, int64_t kvs, int64_t n, int64_t H,
                          int64_t S, float* state, float* scratch, int64_t scratch_floats) try {
    if (n <= 0) return 0;
    if (S <= 0 || S > MAXS || H <= 0) return fail("silo glob: 1 to 8 query rows");
    const int64_t splits = (n + KCH - 1) / KCH;
    if (splits * H * S * ST > scratch_floats) return fail("silo glob: the scratch is too small");
    auto* qu = static_cast<sycl::queue*>(queue);
    const uint16_t* Q = static_cast<const uint16_t*>(qg);
    const uint16_t* K = static_cast<const uint16_t*>(k);
    const uint16_t* Vv = static_cast<const uint16_t*>(v);
    const int s_n = (int) S;
    // 1. per (head, part of KCH keys): the scores' max, the sum of exponentials, the weighted values
    qu->submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> lq(sycl::range<1>(MAXS * D), h);
        sycl::local_accessor<float, 1> lsc(sycl::range<1>(MAXS * KCH), h);
        sycl::local_accessor<float, 1> lacc(sycl::range<1>(4 * MAXS * D), h);
        h.parallel_for(sycl::nd_range<2>(sycl::range<2>((size_t) H, (size_t) (splits * KCH)), sycl::range<2>(1, KCH)), [=](sycl::nd_item<2> it) {
            const int64_t head = it.get_global_id(0), split = it.get_group(1);
            const int lid = (int) it.get_local_id(1);
            auto g = it.get_group();
            for (int i = lid; i < s_n * D; i += KCH) lq[i] = bf(Q[(i / D) * qs + head * D + i % D]) * SCALE;
            sycl::group_barrier(g);
            const int64_t key = split * KCH + lid;
            float sc[MAXS];
            if (key < n) {
                const uint16_t* kr = K + key * kvs + head * D;
                float kv[D];
#pragma unroll
                for (int c = 0; c < D / 8; ++c) {
                    const sycl::vec<uint32_t, 4> w = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(kr + c * 8);
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        kv[c * 8 + 2 * j] = sycl::bit_cast<float>(w[j] << 16);
                        kv[c * 8 + 2 * j + 1] = sycl::bit_cast<float>(w[j] & 0xffff0000u);
                    }
                }
                for (int s = 0; s < s_n; ++s) {
                    float a = 0.0f;
#pragma unroll
                    for (int d = 0; d < D; ++d) a += lq[s * D + d] * kv[d];
                    sc[s] = a;
                }
            } else {
                for (int s = 0; s < s_n; ++s) sc[s] = -INFINITY;
            }
            float m[MAXS], l[MAXS];
            for (int s = 0; s < s_n; ++s) {
                m[s] = sycl::reduce_over_group(g, sc[s], sycl::maximum<float>());
                const float p = (sc[s] == -INFINITY) ? 0.0f : sycl::exp(sc[s] - m[s]);
                lsc[s * KCH + lid] = p;
                l[s] = sycl::reduce_over_group(g, p, sycl::plus<float>());
            }
            sycl::group_barrier(g);
            const int d = lid % D, part = lid / D;     // 4 parts of 64 keys
            float a[MAXS];
            for (int s = 0; s < s_n; ++s) a[s] = 0.0f;
            const int64_t k0 = split * KCH + part * D;
            const int kn = (int) sycl::min<int64_t>(D, n - k0);
            for (int kk = 0; kk < kn; ++kk) {
                const float vv = bf(Vv[(k0 + kk) * kvs + head * D + d]);
                for (int s = 0; s < s_n; ++s) a[s] += lsc[s * KCH + part * D + kk] * vv;
            }
            for (int s = 0; s < s_n; ++s) lacc[(part * MAXS + s) * D + d] = a[s];
            sycl::group_barrier(g);
            if (part == 0) {
                for (int s = 0; s < s_n; ++s) {
                    float* o = scratch + ((split * H + head) * s_n + s) * ST;
                    const float sum = lacc[(0 * MAXS + s) * D + d] + lacc[(1 * MAXS + s) * D + d] + lacc[(2 * MAXS + s) * D + d] +
                                      lacc[(3 * MAXS + s) * D + d];
                    if (d == 0) {
                        o[0] = m[s];
                        o[1] = l[s];
                    }
                    o[2 + d] = sum;
                }
            }
        });
    });
    // 2. the parts folded into the running state, per (head, query row)
    qu->submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (H * S * D)), sycl::range<1>(D)), [=](sycl::nd_item<1> it) {
            const int64_t gi = it.get_group(0);       // head * S + s
            const int64_t head = gi / s_n, s = gi % s_n;
            const int d = (int) it.get_local_id(0);
            float* st = state + gi * ST;
            float M = st[0], L = st[1], A = st[2 + d];
            for (int64_t sp = 0; sp < splits; ++sp) {
                const float* p = scratch + ((sp * H + head) * s_n + s) * ST;
                const float m = p[0];
                const float nm = sycl::fmax(M, m);
                if (nm == -INFINITY) continue;
                const float fa = (M == -INFINITY) ? 0.0f : sycl::exp(M - nm);
                const float fb = (m == -INFINITY) ? 0.0f : sycl::exp(m - nm);
                L = L * fa + p[1] * fb;
                A = A * fa + p[2 + d] * fb;
                M = nm;
            }
            sycl::group_barrier(it.get_group());
            if (d == 0) {
                st[0] = M;
                st[1] = L;
            }
            st[2 + d] = A;
        });
    });
    return 0;
} catch (const std::exception& e) {
    return fail(std::string("silo glob: ") + e.what());
}

int ns_video_h3_silo_glob_out(void* queue, const float* state, int64_t H, int64_t S, void* out, int64_t os) try {
    if (H <= 0 || S <= 0) return 0;
    auto* qu = static_cast<sycl::queue*>(queue);
    uint16_t* O = static_cast<uint16_t*>(out);
    qu->parallel_for(sycl::range<3>((size_t) S, (size_t) H, (size_t) D), [=](sycl::id<3> id) {
        const int64_t s = id[0], head = id[1], d = id[2];
        const float* st = state + (head * S + s) * ST;
        O[s * os + head * D + d] = to_bf(st[1] > 0.0f ? st[2 + d] / st[1] : 0.0f);
    });
    return 0;
} catch (const std::exception& e) {
    return fail(std::string("silo glob_out: ") + e.what());
}

}  // extern "C"
