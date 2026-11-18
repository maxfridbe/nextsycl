// sagebench: MiniMax-H3-Image's SageAttention kernels alone - ARK's (libnextsycl-flash.so's nsflash_sage_square: its
// launch_sage_prefill_kernel_128, half result) and the silo's (kernels/image/h3image/silo/tla/sage.cpp, compiled in
// here: bfloat16 result in [S, H * D]) - on the same int8 inputs, timed (median of ITERS) and checked against float32
// attention: of the dequantized int8 q / k and half v (what the kernel should compute) and of the float q / k / v.
// ARK's half result is rounded to bfloat16 first, as the silo's conversion pass did, so both are judged as the engine
// uses them.
//   sagebench S H ITERS [KERNELS...]   (0 ARK, 1 silo; default both; repeat an index to alternate)
// Build and run in the build image (sycl-tla's flags, as kernels/build.sh builds the tla silos), one card:
//   icpx -O3 -fsycl -std=c++17 -fno-sycl-instrument-device-code -w -DARK_XPU=1 -DARK_SYCL_TLA=1 -DCUTLASS_ENABLE_SYCL=1
//     -DSYCL_INTEL_TARGET=1 -isystem $TLA/include -isystem $TLA/applications -isystem $TLA/tools/util/include
//     -isystem $TLA/examples/common -isystem $TLA/examples/06_bmg_flash_attention -I$ARK/wrapper/include -I$ARK/bestla
//     -Ikernels/image/h3image/silo/tla reference/h3-image/sagebench.cpp -fsycl-targets=spir64_gen
//     -Xsycl-target-backend=spir64_gen "-device bmg-g31" -Xspirv-translator
//     -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate -o sagebench
//   ONEAPI_DEVICE_SELECTOR=level_zero:0 ./sagebench 16434 56 20
// P-H3-Image's copy of the kernel (kernels/image/ph3image/silo/tla) instead: -Ikernels/image/ph3image/silo/tla
//   -Dns_image_h3i_tla_sage=ns_image_ph3_tla_sage (the same code under its own names; "silo" then means P-H3-Image's)
#include <sycl/sycl.hpp>
#include "sage.cpp"   // the silo's kernel (kernels/image/h3image/silo/tla/sage.cpp), with ARK's headers

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace cute;
using ark::detail::Options;

static uint16_t* g_ob = nullptr;   // the silo kernel's bfloat16 result
static sycl::queue* g_q = nullptr;

static int run_ark(const Options& o) {
    return ark::detail::launch_sage_prefill_kernel_128<cute::int8_t, cute::int8_t, cute::half_t>(o);
}
static int run_silo(const Options& o) {
    return ns_image_h3i_tla_sage(g_q, (const int8_t*) o.q, (const int8_t*) o.k, o.v, g_ob, (const float*) o.qscale,
                                 (const float*) o.kscale, o.scale_block_size, o.seq_len_qo, o.num_heads_q, o.head_size_qk, o.softmax_scale);
}
struct Cfg { const char* name; int (*fn)(const Options&); bool t = false; };
static Cfg cfgs[] = {
    {"ARK (half result, then bfloat16)", run_ark},
    {"silo (bfloat16 result)", run_silo, true},
};
constexpr int NCFG = sizeof(cfgs) / sizeof(cfgs[0]);

int main(int argc, char** argv) {
    const int S = argc > 1 ? atoi(argv[1]) : 16434, H = argc > 2 ? atoi(argv[2]) : 56, D = 128, iters = argc > 3 ? atoi(argv[3]) : 10;
    std::vector<int> pick;
    for (int i = 4; i < argc; ++i) pick.push_back(atoi(argv[i]));
    if (pick.empty()) for (int i = 0; i < NCFG; ++i) pick.push_back(i);
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
    std::fprintf(stdout, "device %s, S %d H %d D %d\n", q.get_device().get_info<sycl::info::device::name>().c_str(), S, H, D);
    compat::set_default_queue(q);
    g_q = &q;
    const size_t n = (size_t) S * H * D;
    const int BLK = 64, nb = (S + BLK - 1) / BLK;
    // floats [H, S, D]: q, k ~ N(0, 1.6) plus a per-column k offset, v ~ N(0, 1)
    std::vector<float> fq(n), fk(n), fv(n);
    {
        std::mt19937 rng(7);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::vector<float> koff((size_t) H * D);
        for (auto& x : koff) x = nd(rng) * 2.0f;
        for (size_t i = 0; i < n; ++i) {
            fq[i] = nd(rng) * 1.6f;
            fk[i] = nd(rng) * 1.6f + koff[(i / ((size_t) S * D)) * D + i % D];
            fv[i] = nd(rng);
        }
    }
    // the silo's preparation: k less its mean over the sequence; q, k int8 a scale per (head, 64 rows); v half
    std::vector<int8_t> iq(n), ik(n);
    std::vector<float> qs((size_t) H * nb), ks((size_t) H * nb);
    std::vector<sycl::half> hv(n);
    std::vector<float> dq(n), dk(n);   // the int8 values times their scales (the kernel's exact inputs)
    for (int h = 0; h < H; ++h) {
        std::vector<double> mean(D, 0.0);
        for (int s = 0; s < S; ++s) for (int d = 0; d < D; ++d) mean[d] += fk[((size_t) h * S + s) * D + d];
        for (int s = 0; s < S; ++s) for (int d = 0; d < D; ++d) fk[((size_t) h * S + s) * D + d] -= (float) (mean[d] / S);
        for (int b = 0; b < nb; ++b) {
            for (int t = 0; t < 2; ++t) {
                const std::vector<float>& f = t ? fk : fq;
                float m = 0;
                for (int s = b * BLK; s < std::min(S, (b + 1) * BLK); ++s) for (int d = 0; d < D; ++d) m = std::max(m, std::fabs(f[((size_t) h * S + s) * D + d]));
                const float inv = m > 0 ? 127.0f / m : 0.0f;
                (t ? ks : qs)[(size_t) h * nb + b] = m / 127.0f;
                for (int s = b * BLK; s < std::min(S, (b + 1) * BLK); ++s) for (int d = 0; d < D; ++d) {
                    const size_t i = ((size_t) h * S + s) * D + d;
                    const int8_t v = (int8_t) std::max(-127.0f, std::min(127.0f, std::round(f[i] * inv)));
                    (t ? ik : iq)[i] = v;
                    (t ? dk : dq)[i] = v * (m / 127.0f);
                }
            }
        }
    }
    for (size_t i = 0; i < n; ++i) hv[i] = (sycl::half) fv[i];
    int8_t* d_q = sycl::malloc_device<int8_t>(n, q);
    int8_t* d_k = sycl::malloc_device<int8_t>(n, q);
    sycl::half* d_v = sycl::malloc_device<sycl::half>(n, q);
    sycl::half* d_o = sycl::malloc_device<sycl::half>(n, q);
    float* d_qs = sycl::malloc_device<float>(qs.size(), q);
    float* d_ks = sycl::malloc_device<float>(ks.size(), q);
    float* d_fq = sycl::malloc_device<float>(n, q);
    float* d_fk = sycl::malloc_device<float>(n, q);
    float* d_fv = sycl::malloc_device<float>(n, q);
    float* d_ref = sycl::malloc_device<float>(n, q);
    q.memcpy(d_q, iq.data(), n).wait();
    q.memcpy(d_k, ik.data(), n).wait();
    q.memcpy(d_v, hv.data(), n * 2).wait();
    q.memcpy(d_qs, qs.data(), qs.size() * 4).wait();
    q.memcpy(d_ks, ks.data(), ks.size() * 4).wait();
    const float scale = 1.0f / std::sqrt((float) D);
    // float32 references: (a) of the int8 inputs dequantized and v in half (the kernel's own math), (b) of the floats
    auto reference = [&](const std::vector<float>& rq, const std::vector<float>& rk, const std::vector<float>& rv, std::vector<float>& out) {
        q.memcpy(d_fq, rq.data(), n * 4).wait();
        q.memcpy(d_fk, rk.data(), n * 4).wait();
        q.memcpy(d_fv, rv.data(), n * 4).wait();
        const int RB = 8;   // rows a work-group; 128 work-items, one an output column
        const int NG = (S + RB - 1) / RB;
        q.submit([&](sycl::handler& h) {
            sycl::local_accessor<float, 1> kt(sycl::range<1>(64 * 129), h), vt(sycl::range<1>(64 * 128), h), qt(sycl::range<1>(RB * 128), h),
                p(sycl::range<1>(RB * 64), h), st(sycl::range<1>(3 * RB), h);
            h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) H * NG * 128), sycl::range<1>(128)), [=](sycl::nd_item<1> it) {
                const int g = (int) it.get_group(0), hh = g / NG, r0 = (g % NG) * RB;
                const int lid = (int) it.get_local_id(0);
                for (int r = 0; r < RB; ++r) qt[r * 128 + lid] = r0 + r < S ? d_fq[((size_t) hh * S + r0 + r) * D + lid] : 0.0f;
                if (lid < RB) { st[lid] = -INFINITY; st[RB + lid] = 0.0f; }
                float acc[RB];
                for (int r = 0; r < RB; ++r) acc[r] = 0.0f;
                for (int c0 = 0; c0 < S; c0 += 64) {
                    const int cn = sycl::min(64, S - c0);
                    for (int i = lid; i < 64 * 128; i += 128) {
                        const int c = i / 128, d = i % 128;
                        kt[c * 129 + d] = c < cn ? d_fk[((size_t) hh * S + c0 + c) * D + d] : 0.0f;
                        vt[c * 128 + d] = c < cn ? d_fv[((size_t) hh * S + c0 + c) * D + d] : 0.0f;
                    }
                    sycl::group_barrier(it.get_group());
                    for (int i = lid; i < RB * 64; i += 128) {
                        const int r = i / 64, c = i % 64;
                        float a = 0.0f;
                        for (int d = 0; d < 128; ++d) a += qt[r * 128 + d] * kt[c * 129 + d];
                        p[i] = c < cn ? a * scale : -INFINITY;
                    }
                    sycl::group_barrier(it.get_group());
                    if (lid < RB) {
                        float bm = -INFINITY;
                        for (int c = 0; c < 64; ++c) bm = sycl::max(bm, p[lid * 64 + c]);
                        const float om = st[lid], nm = sycl::max(om, bm);
                        st[2 * RB + lid] = sycl::exp(om - nm);
                        st[lid] = nm;
                    }
                    sycl::group_barrier(it.get_group());
                    for (int i = lid; i < RB * 64; i += 128) p[i] = sycl::exp(p[i] - st[i / 64]);
                    sycl::group_barrier(it.get_group());
                    if (lid < RB) {
                        float ls = 0.0f;
                        for (int c = 0; c < 64; ++c) ls += p[lid * 64 + c];
                        st[RB + lid] = st[RB + lid] * st[2 * RB + lid] + ls;
                    }
                    for (int r = 0; r < RB; ++r) {
                        float a = acc[r] * st[2 * RB + r];
                        for (int c = 0; c < cn; ++c) a += p[r * 64 + c] * vt[c * 128 + lid];
                        acc[r] = a;
                    }
                    sycl::group_barrier(it.get_group());
                }
                for (int r = 0; r < RB; ++r) if (r0 + r < S) d_ref[((size_t) hh * S + r0 + r) * D + lid] = acc[r] / st[RB + r];
            });
        }).wait();
        out.resize(n);
        q.memcpy(out.data(), d_ref, n * 4).wait();
    };
    std::vector<float> hvf(n);
    for (size_t i = 0; i < n; ++i) hvf[i] = (float) hv[i];
    std::vector<float> ref_q, ref_f;
    auto t0 = std::chrono::steady_clock::now();
    reference(dq, dk, hvf, ref_q);
    reference(fq, fk, fv, ref_f);
    std::fprintf(stdout, "references %.1f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    Options o;
    o.q = d_q; o.k = d_k; o.v = d_v; o.o = d_o;
    o.scale_block_size = BLK; o.qscale = d_qs; o.kscale = d_ks;
    o.batch = 1; o.num_heads_q = o.num_heads_kv = H; o.seq_len_qo = o.seq_len_kv = S;
    o.head_size_qk = o.head_size_vo = D; o.softmax_scale = scale; o.is_causal = false;
    const double flop = 4.0 * S * (double) S * D * H;
    std::vector<sycl::half> out(n);
    uint16_t* d_ob = sycl::malloc_device<uint16_t>(n, q);
    g_ob = d_ob;
    std::vector<uint16_t> ob(n);
    for (int c : pick) {
        q.memset(d_o, 0, n * 2).wait();
        q.memset(d_ob, 0, n * 2).wait();
        int rc = 0;
        try { rc = cfgs[c].fn(o); q.wait(); } catch (const std::exception& e) { std::fprintf(stdout, "[%d] %s: %s\n", c, cfgs[c].name, e.what()); continue; }
        if (rc) { std::fprintf(stdout, "[%d] %s: rc %d\n", c, cfgs[c].name, rc); continue; }
        if (cfgs[c].t) {
            q.memcpy(ob.data(), d_ob, n * 2).wait();
            for (int h = 0; h < H; ++h) for (int s2 = 0; s2 < S; ++s2) for (int d = 0; d < D; ++d) {
                uint32_t u = (uint32_t) ob[((size_t) s2 * H + h) * D + d] << 16;
                float f; std::memcpy(&f, &u, 4);
                out[((size_t) h * S + s2) * D + d] = (sycl::half) f;
            }
        } else {
            q.memcpy(out.data(), d_o, n * 2).wait();
            for (size_t i = 0; i < n; ++i) {
                float f = (float) out[i];
                uint32_t u;
                std::memcpy(&u, &f, 4);
                u += 0x7fffu + ((u >> 16) & 1u);
                u &= 0xffff0000u;
                std::memcpy(&f, &u, 4);
                out[i] = (sycl::half) f;
            }
        }
        double e1 = 0, m1 = 0, e2 = 0, m2 = 0, r2 = 0, rn = 0;
        size_t bad = 0;
        for (size_t i = 0; i < n; ++i) {
            const double v = (double) (float) out[i];
            if (!std::isfinite(v)) { ++bad; continue; }
            const double a = std::fabs(v - ref_q[i]), b = std::fabs(v - ref_f[i]);
            e1 += a; m1 = std::max(m1, a); e2 += b; m2 = std::max(m2, b);
            r2 += (v - ref_f[i]) * (v - ref_f[i]); rn += ref_f[i] * (double) ref_f[i];
        }
        for (int w = 0; w < 2; ++w) cfgs[c].fn(o);
        q.wait();
        std::vector<double> ms;
        for (int it = 0; it < iters; ++it) {
            auto a = std::chrono::steady_clock::now();
            cfgs[c].fn(o);
            q.wait();
            ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count());
        }
        std::sort(ms.begin(), ms.end());
        const double med = ms[ms.size() / 2];
        std::fprintf(stdout, "[%d] %-44s %8.3f ms (min %.3f) %6.1f TFLOPS | vs int8-math ref: max %.3e mean %.3e | vs float ref: max %.3e mean %.3e rel %.3e | nonfinite %zu\n",
                    c, cfgs[c].name, med, ms[0], flop / med / 1e9, m1, e1 / n, m2, e2 / n, std::sqrt(r2 / rn), bad);
        std::fflush(stdout);
    }
    return 0;
}
