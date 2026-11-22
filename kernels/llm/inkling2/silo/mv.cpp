// mv.cpp: the inkling2 silo's decode products (silo.h). The Q8_0 and NVFP4 matrices are repacked at load (the quants or
// codes of the rows apart from their scales), so a lane reads a 16-byte line a step and a sub-group's lanes read 256
// consecutive bytes: a Q8_0 line is 16 quants (half a block), an NVFP4 line 32 codes (a group of two sub-blocks). The
// activations are quantized as ggml's Q8_0; a line is an int8 dot (dp4a) against them, scaled by the weight's and the
// activation's block scales. A sub-group takes R rows (the activations' line loaded once for them), its lanes the lines
// in turn; the lanes' sums are added in a fixed order, so a row's value is the same whatever rows share a pass. The
// routed experts: gate | up and SiTU in one launch (the act quantized for the down products in the work-group), the
// down products of a row's K experts and their weighted sum in another.
#include "ns.h"
#include "ns_internal.hpp"
#include "silo.h"

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cstring>
#include <type_traits>

namespace {
using half = sycl::half;
constexpr int SG = 16;
constexpr int MAXT = 16;

inline size_t align256(size_t n) { return (n + 255) / 256 * 256; }

// ---- the formats ----
inline float ue4m3(uint32_t x) {
    x &= 0xFF;
    if (x == 0 || x == 0x7F) return 0.0f;
    const int e = (int) ((x >> 3) & 0xF), m = (int) (x & 0x7);
    return e == 0 ? sycl::ldexp((float) m, -10) : sycl::ldexp((float) (8 + m), e - 11);
}

// a code byte's two E2M1 values (doubled: 0 1 2 3 4 6 8 12, the top bit of a code its sign) as signed bytes: the low
// nibble's in bits 0-7, the high nibble's in bits 16-23 (the decode table, 256 words in local memory)
inline uint32_t fp4_entry(uint32_t b) {
    auto val = [](uint32_t c) {
        int v = (int) ((0xC8643210u >> ((c & 7u) << 2)) & 15u);
        return (uint32_t) ((c & 8u) ? -v : v) & 0xFFu;
    };
    return val(b & 15u) | (val(b >> 4) << 16);
}

// four code bytes (c: bytes 0-3 of a sub-block) to their eight values as two words of signed bytes: lo the low nibbles'
// (values j), hi the high nibbles' (values j + 8)
inline void fp4_words(uint32_t c, const uint32_t* lut, int& lo, int& hi) {
    const uint32_t a = lut[c & 0xFF] | (lut[(c >> 8) & 0xFF] << 8);
    const uint32_t b = lut[(c >> 16) & 0xFF] | (lut[c >> 24] << 8);
    lo = (int) ((a & 0xFFFFu) | (b << 16));
    hi = (int) ((a >> 16) | (b & 0xFFFF0000u));
}

inline int dp4a(int a, int b, int c) { return dpct::dp4a(a, b, c); }

// a repacked matrix's lines: Q8_0 16 quants (n / 16 a row), NVFP4 32 codes (n / 32 a row); a row's quant bytes
template <int TY>
constexpr int64_t lines(int64_t n) { return TY == 8 ? n / 16 : n / 32; }
inline int64_t qrow(int type, int64_t n) { return type == 8 ? n : n / 2; }
inline int64_t srow(int type, int64_t n) { return type == 8 ? (n / 32) * 2 : n / 16; }

// line li of a row: its words and factors, against the activations' line (Q8_0: 16 values, 4 words, one scale;
// NVFP4: 32 values, 8 words, two scales)
template <int TY>
struct Line {
    int w[TY == 8 ? 4 : 8];
    float f[TY == 8 ? 1 : 2];
};

template <int TY>
inline Line<TY> load_line(const int8_t* q, const uint8_t* s, int64_t li, const uint32_t* lut) {
    Line<TY> l;
    const sycl::int4 v = *(const sycl::int4*) (q + li * 16);
    if constexpr (TY == 8) {
        l.w[0] = v.x(); l.w[1] = v.y(); l.w[2] = v.z(); l.w[3] = v.w();
        uint16_t h;
        std::memcpy(&h, s + (li >> 1) * 2, 2);
        l.f[0] = (float) sycl::bit_cast<half>(h);
    } else {
        // the group's 16 bytes: sub-block 0's 8 then sub-block 1's; a byte's low nibble value j, its high one j + 8
        const uint32_t c0 = (uint32_t) v.x(), c1 = (uint32_t) v.y(), c2 = (uint32_t) v.z(), c3 = (uint32_t) v.w();
        fp4_words(c0, lut, l.w[0], l.w[2]);
        fp4_words(c1, lut, l.w[1], l.w[3]);
        fp4_words(c2, lut, l.w[4], l.w[6]);
        fp4_words(c3, lut, l.w[5], l.w[7]);
        uint16_t sc;
        std::memcpy(&sc, s + li * 2, 2);
        l.f[0] = ue4m3(sc);
        l.f[1] = ue4m3(sc >> 8);
    }
    return l;
}

// the activations' line li of a row (their quants q, scales d)
template <int TY>
struct XLine {
    int x[TY == 8 ? 4 : 8];
    float d;
};

template <int TY>
inline XLine<TY> load_x(const int8_t* q, const float* d, int64_t li) {
    XLine<TY> l;
    if constexpr (TY == 8) {
        const sycl::int4 v = *(const sycl::int4*) (q + li * 16);
        l.x[0] = v.x(); l.x[1] = v.y(); l.x[2] = v.z(); l.x[3] = v.w();
        l.d = d[li >> 1];
    } else {
        const sycl::int4* p = (const sycl::int4*) (q + li * 32);
        const sycl::int4 a = p[0], b = p[1];
        l.x[0] = a.x(); l.x[1] = a.y(); l.x[2] = a.z(); l.x[3] = a.w();
        l.x[4] = b.x(); l.x[5] = b.y(); l.x[6] = b.z(); l.x[7] = b.w();
        l.d = d[li];
    }
    return l;
}

// a line's sum added to acc, the same operations in every kernel and row count (explicit fma: no contraction left to
// the compiler)
template <int TY>
inline float ldot(float acc, const Line<TY>& w, const XLine<TY>& x) {
    if constexpr (TY == 8) {
        int s = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) s = dp4a(w.w[i], x.x[i], s);
        return sycl::fma((float) s, w.f[0] * x.d, acc);
    } else {
        int s0 = 0, s1 = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) s0 = dp4a(w.w[i], x.x[i], s0);
#pragma unroll
        for (int i = 4; i < 8; ++i) s1 = dp4a(w.w[i], x.x[i], s1);
        return sycl::fma(sycl::fma((float) s0, w.f[0], (float) s1 * w.f[1]), x.d, acc);
    }
}

// the decode table in local memory, filled by the work-group
inline void fill_lut(sycl::nd_item<1> it, uint32_t* lut) {
    for (int i = (int) it.get_local_id(0); i < 256; i += (int) it.get_local_range(0)) lut[i] = fp4_entry((uint32_t) i);
    sycl::group_barrier(it.get_group());
}

// ---- dense products: R rows a sub-group, NSG sub-groups a work-group ----
constexpr int R = 4, NSG = 4;

struct Job {
    const int8_t* q;    // the matrix's quants
    const uint8_t* s;   // its scales
    const int8_t* xq;
    const float* xd;
    float* y;
    int64_t rows, n_in;
    int64_t blk0;       // its first row block in the launch
};
struct Jobs {
    Job j[8];
    int n;
};

template <int TY, int TT>
void mv_launch(sycl::queue& q, Jobs jobs, int64_t blocks) {
    q.submit([&](sycl::handler& h) {
    sycl::local_accessor<uint32_t, 1> lutm(sycl::range<1>(256), h);
    h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) blocks * SG), sycl::range<1>(NSG * SG)), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        uint32_t* lut = lutm.get_multi_ptr<sycl::access::decorated::no>().get();
        if constexpr (TY == 40) fill_lut(it, lut);
        auto sg = it.get_sub_group();
        const int lane = (int) sg.get_local_linear_id();
        const int64_t blk = it.get_group(0) * NSG + sg.get_group_linear_id();
        int ji = 0;
#pragma unroll
        for (int i = 1; i < 8; ++i)
            if (i < jobs.n && blk >= jobs.j[i].blk0) ji = i;
        const Job jb = jobs.j[ji];
        const int64_t r0 = (blk - jb.blk0) * R;
        if (r0 >= jb.rows) return;
        const int64_t nl = lines<TY>(jb.n_in), qr = qrow(TY, jb.n_in), sr = srow(TY, jb.n_in), nb = jb.n_in / 32;
        int nr = (int) sycl::min<int64_t>(R, jb.rows - r0);
        float acc[R][TT];
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int t = 0; t < TT; ++t) acc[r][t] = 0.0f;
        for (int64_t li = lane; li < nl; li += SG) {
            Line<TY> w[R];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const int64_t row = r0 + sycl::min(r, nr - 1);
                w[r] = load_line<TY>(jb.q + row * qr, jb.s + row * sr, li, lut);
            }
#pragma unroll
            for (int t = 0; t < TT; ++t) {
                const XLine<TY> x = load_x<TY>(jb.xq + t * jb.n_in, jb.xd + t * nb, li);
#pragma unroll
                for (int r = 0; r < R; ++r) acc[r][t] = ldot<TY>(acc[r][t], w[r], x);
            }
        }
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int t = 0; t < TT; ++t) {
                const float v = sycl::reduce_over_group(sg, acc[r][t], sycl::plus<float>());
                if (lane == 0 && r < nr) jb.y[t * jb.rows + r0 + r] = v;
            }
    });
    });
}

template <int TY>
void mv_t(sycl::queue& q, const Jobs& jobs, int64_t blocks, int64_t T) {
    switch (T) {
        case 1: mv_launch<TY, 1>(q, jobs, blocks); break;
        case 2: mv_launch<TY, 2>(q, jobs, blocks); break;
        case 3: mv_launch<TY, 3>(q, jobs, blocks); break;
        case 4: mv_launch<TY, 4>(q, jobs, blocks); break;
        case 5: mv_launch<TY, 5>(q, jobs, blocks); break;
        case 6: mv_launch<TY, 6>(q, jobs, blocks); break;
        case 7: mv_launch<TY, 7>(q, jobs, blocks); break;
        case 8: mv_launch<TY, 8>(q, jobs, blocks); break;
        case 9: mv_launch<TY, 9>(q, jobs, blocks); break;
        case 10: mv_launch<TY, 10>(q, jobs, blocks); break;
        case 11: mv_launch<TY, 11>(q, jobs, blocks); break;
        case 12: mv_launch<TY, 12>(q, jobs, blocks); break;
        case 13: mv_launch<TY, 13>(q, jobs, blocks); break;
        case 14: mv_launch<TY, 14>(q, jobs, blocks); break;
        case 15: mv_launch<TY, 15>(q, jobs, blocks); break;
        default: mv_launch<TY, 16>(q, jobs, blocks); break;
    }
}

// ggml's Q8_0 rounding of a block of 32 (16 lanes, 2 values each): d = amax / 127 (kept as half), q = round(x / d)
inline void quant_block(sycl::sub_group sg, int lane, const float* x, int8_t* q, float* d) {
    const float a = x[lane * 2], b = x[lane * 2 + 1];
    const float amax = sycl::reduce_over_group(sg, sycl::fmax(sycl::fabs(a), sycl::fabs(b)), sycl::maximum<float>());
    const float dd = amax / 127.0f;
    const float id = dd != 0.0f ? 1.0f / dd : 0.0f;
    q[lane * 2] = (int8_t) sycl::round(a * id);
    q[lane * 2 + 1] = (int8_t) sycl::round(b * id);
    if (lane == 0) *d = (float) half(dd);
}

inline const int8_t* xq_q(const void* xq) { return (const int8_t*) xq; }
inline const float* xq_d(const void* xq, int64_t T, int64_t n) { return (const float*) ((const uint8_t*) xq + ns_ink2s_xq_scales_at(T, n)); }

}  // namespace

extern "C" {

int ns_ink2s_supported(int type, int64_t n_in) {
    return (type == 8 && n_in % 32 == 0) || (type == 40 && n_in % 64 == 0);
}

size_t ns_ink2s_scales_at(int type, int64_t rows, int64_t n_in) {
    return align256((size_t) (rows * qrow(type, n_in)));
}
size_t ns_ink2s_packed_bytes(int type, int64_t rows, int64_t n_in) {
    return ns_ink2s_scales_at(type, rows, n_in) + align256((size_t) (rows * srow(type, n_in)));
}

int ns_ink2s_pack(ns_gpu* g, int type, const void* raw, int64_t rows, int64_t n_in, void* dst) {
    NS_TRY
    if (!ns_ink2s_supported(type, n_in)) return ns_fail("ns_ink2s_pack: a type or width it does not take");
    const uint8_t* src = (const uint8_t*) raw;
    int8_t* q = (int8_t*) dst;
    uint8_t* s = (uint8_t*) dst + ns_ink2s_scales_at(type, rows, n_in);
    if (type == 8) {
        // a block: its half scale, its 32 quants
        const int64_t nb = rows * (n_in / 32);
        g->q.parallel_for(sycl::range<1>((size_t) nb), [=](sycl::id<1> id) {
            const int64_t b = id[0];
            const uint8_t* p = src + b * 34;
            s[b * 2] = p[0];
            s[b * 2 + 1] = p[1];
            for (int i = 0; i < 32; ++i) q[b * 32 + i] = (int8_t) p[2 + i];
        });
    } else {
        // a block of 64: its four scales, its 32 code bytes (two groups of 16)
        const int64_t nb = rows * (n_in / 64);
        g->q.parallel_for(sycl::range<1>((size_t) nb), [=](sycl::id<1> id) {
            const int64_t b = id[0];
            const uint8_t* p = src + b * 36;
            for (int i = 0; i < 4; ++i) s[b * 4 + i] = p[i];
            for (int i = 0; i < 32; ++i) q[b * 32 + i] = (int8_t) p[4 + i];
        });
    }
    return 0;
    NS_CATCH
}

int ns_ink2s_unpack_f16(ns_gpu* g, int type, const void* w, int64_t rows, int64_t row0, int64_t n, int64_t n_in, uint16_t* dst) {
    NS_TRY
    if (!ns_ink2s_supported(type, n_in) || row0 + n > rows) return ns_fail("ns_ink2s_unpack_f16: a type, width or rows it does not take");
    const int8_t* q = (const int8_t*) w;
    const uint8_t* s = (const uint8_t*) w + ns_ink2s_scales_at(type, rows, n_in);
    const int64_t qr = qrow(type, n_in), sr = srow(type, n_in);
    // a work-item a group of 16 values
    const int64_t per = n_in / 16, total = n * per;
    if (total <= 0) return 0;
    if (type == 8) {
        g->q.parallel_for(sycl::range<1>((size_t) total), [=](sycl::id<1> id) {
            const int64_t r = id[0] / per, c = id[0] % per, row = row0 + r;
            uint16_t h;
            std::memcpy(&h, s + row * sr + (c >> 1) * 2, 2);
            const float d = (float) sycl::bit_cast<half>(h);
            const int8_t* qq = q + row * qr + c * 16;
            for (int i = 0; i < 16; ++i) dst[r * n_in + c * 16 + i] = sycl::bit_cast<uint16_t>(half(d * (float) qq[i]));
        });
    } else {
        g->q.parallel_for(sycl::range<1>((size_t) total), [=](sycl::id<1> id) {
            const int64_t r = id[0] / per, c = id[0] % per, row = row0 + r;
            // sub-block c: its 8 code bytes, its scale
            const float d = ue4m3(s[row * sr + c]);
            const uint8_t* cb = (const uint8_t*) q + row * qr + c * 8;
            constexpr float t[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
            for (int j = 0; j < 8; ++j) {
                dst[r * n_in + c * 16 + j] = sycl::bit_cast<uint16_t>(half(t[cb[j] & 15] * d));
                dst[r * n_in + c * 16 + j + 8] = sycl::bit_cast<uint16_t>(half(t[cb[j] >> 4] * d));
            }
        });
    }
    return 0;
    NS_CATCH
}

size_t ns_ink2s_xq_scales_at(int64_t T, int64_t n) {
    return align256((size_t) (T * n));
}
size_t ns_ink2s_xq_bytes(int64_t T, int64_t n) {
    return ns_ink2s_xq_scales_at(T, n) + align256((size_t) (T * (n / 32)) * 4);
}

int ns_ink2s_xq(ns_gpu* g, const float* x, void* xq, int64_t T, int64_t n) {
    NS_TRY
    if (n % 32) return ns_fail("ns_ink2s_xq: rows of whole blocks of 32");
    const int64_t nb = T * (n / 32);
    int8_t* q = (int8_t*) xq;
    float* d = (float*) ((uint8_t*) xq + ns_ink2s_xq_scales_at(T, n));
    if (nb > 0) g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) ((nb + 3) / 4) * 4 * SG), sycl::range<1>(4 * SG)),
                                  [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        auto sg = it.get_sub_group();
        const int64_t b = it.get_group(0) * 4 + sg.get_group_linear_id();
        if (b >= nb) return;
        quant_block(sg, (int) sg.get_local_linear_id(), x + b * 32, q + b * 32, d + b);
    });
    return 0;
    NS_CATCH
}

int ns_ink2s_rms_xq(ns_gpu* g, const float* x, const float* w, float* y, void* xq, int64_t T, int64_t n, float eps) {
    NS_TRY
    if (n % 32 || n > 4096) return ns_fail("ns_ink2s_rms_xq: rows of whole blocks of 32, at most 4096");
    constexpr int W = 256;
    int8_t* q = (int8_t*) xq;
    float* d = (float*) ((uint8_t*) xq + ns_ink2s_xq_scales_at(T, n));
    if (T > 0) g->q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> row(sycl::range<1>(4096), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) T * W), sycl::range<1>(W)), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const int64_t t = it.get_group(0);
            const int tid = (int) it.get_local_id(0);
            const float* xr = x + t * n;
            float ss = 0.0f;
            for (int64_t i = tid; i < n; i += W) ss += xr[i] * xr[i];
            const float inv = sycl::rsqrt(sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>()) / (float) n + eps);
            for (int64_t i = tid; i < n; i += W) {
                const float v = xr[i] * inv * w[i];
                row[i] = v;
                if (y) y[t * n + i] = v;
            }
            sycl::group_barrier(it.get_group());
            auto sg = it.get_sub_group();
            const int lane = (int) sg.get_local_linear_id(), sgi = (int) sg.get_group_linear_id();
            float* rp = row.get_multi_ptr<sycl::access::decorated::no>().get();
            for (int64_t b = sgi; b < n / 32; b += W / SG) quant_block(sg, lane, rp + b * 32, q + t * n + b * 32, d + t * (n / 32) + b);
        });
    });
    return 0;
    NS_CATCH
}

int ns_ink2s_mv_many(ns_gpu* g, int n, const int* ty, const void* const* w, const void* const* xq, float* const* y, const int64_t* rows,
                     const int64_t* n_in, int64_t T) {
    NS_TRY
    if (n < 1 || n > 8 || T < 1 || T > MAXT) return ns_fail("ns_ink2s_mv_many: 1..8 products, 1..16 rows");
    // the jobs of one type a launch (the launches in order)
    for (int i = 0; i < n;) {
        Jobs js{};
        int64_t blocks = 0;
        const int t0 = ty[i];
        for (; i < n && ty[i] == t0 && js.n < 8; ++i) {
            if (!ns_ink2s_supported(ty[i], n_in[i])) return ns_fail("ns_ink2s_mv_many: a type or width it does not take");
            Job& j = js.j[js.n++];
            j.q = (const int8_t*) w[i];
            j.s = (const uint8_t*) w[i] + ns_ink2s_scales_at(ty[i], rows[i], n_in[i]);
            j.xq = xq_q(xq[i]);
            j.xd = xq_d(xq[i], T, n_in[i]);
            j.y = y[i];
            j.rows = rows[i];
            j.n_in = n_in[i];
            j.blk0 = blocks;
            blocks += (rows[i] + R - 1) / R;
        }
        const int64_t wgs = (blocks + NSG - 1) / NSG;
        if (t0 == 8) mv_t<8>(g->q, js, wgs * NSG, T);
        else mv_t<40>(g->q, js, wgs * NSG, T);
    }
    return 0;
    NS_CATCH
}

int ns_ink2s_mv(ns_gpu* g, int type, const void* w, const void* xq, float* y, int64_t n_in, int64_t rows, int64_t T) {
    return ns_ink2s_mv_many(g, 1, &type, &w, &xq, &y, &rows, &n_in, T);
}

int ns_ink2s_moe_gu(ns_gpu* g, int type, const void* w, int64_t E, const float* scale, const int32_t* ids, const void* xq, void* act_q, int64_t T,
                    int64_t K, int64_t n_in, int64_t F, float beta, float lb) {
    NS_TRY
    if (!ns_ink2s_supported(type, n_in) || F % 32) return ns_fail("ns_ink2s_moe_gu: a type or shape it does not take");
    // a work-group a (pair, 32 act values): 8 sub-groups of 4 values, each 4 gate and 4 up rows
    constexpr int NS8 = 8, RV = 4;
    const int64_t P = T * K, nblk = F / 32, nl = type == 8 ? n_in / 16 : n_in / 32, nbx = n_in / 32;
    const int64_t rows = E * 2 * F, qr = qrow(type, n_in), sr = srow(type, n_in);
    const int8_t* wq = (const int8_t*) w;
    const uint8_t* ws = (const uint8_t*) w + ns_ink2s_scales_at(type, rows, n_in);
    const int8_t* xqq = xq_q(xq);
    const float* xqd = xq_d(xq, T, n_in);
    int8_t* aq = (int8_t*) act_q;
    float* ad = (float*) ((uint8_t*) act_q + ns_ink2s_xq_scales_at(P, F));
    auto run = [&](auto tyc) {
        constexpr int TY = decltype(tyc)::value;
        g->q.submit([&](sycl::handler& h) {
            sycl::local_accessor<float, 1> av(sycl::range<1>(32), h);
            sycl::local_accessor<uint32_t, 1> lutm(sycl::range<1>(256), h);
            h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (P * nblk) * NS8 * SG), sycl::range<1>(NS8 * SG)),
                           [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                uint32_t* lut = lutm.get_multi_ptr<sycl::access::decorated::no>().get();
                if constexpr (TY == 40) fill_lut(it, lut);
                auto sg = it.get_sub_group();
                const int lane = (int) sg.get_local_linear_id(), sgi = (int) sg.get_group_linear_id();
                const int64_t p = it.get_group(0) / nblk, ab = it.get_group(0) % nblk;
                const int64_t t = p / K, e = ids[p];
                const int64_t base = e * 2 * F, i0 = ab * 32 + sgi * RV;
                float acc[2 * RV];
#pragma unroll
                for (int r = 0; r < 2 * RV; ++r) acc[r] = 0.0f;
                for (int64_t li = lane; li < nl; li += SG) {
                    const XLine<TY> x = load_x<TY>(xqq + t * n_in, xqd + t * nbx, li);
#pragma unroll
                    for (int r = 0; r < 2 * RV; ++r) {
                        const int64_t row = base + (r < RV ? i0 + r : F + i0 + (r - RV));
                        acc[r] = ldot<TY>(acc[r], load_line<TY>(wq + row * qr, ws + row * sr, li, lut), x);
                    }
                }
                const float s = scale ? scale[e] : 1.0f;
#pragma unroll
                for (int r = 0; r < RV; ++r) {
                    const float gt = sycl::reduce_over_group(sg, acc[r], sycl::plus<float>()) * s;
                    const float up = sycl::reduce_over_group(sg, acc[RV + r], sycl::plus<float>()) * s;
                    const float a = beta * sycl::tanh(gt / beta) * (1.0f / (1.0f + sycl::exp(-gt)));
                    const float u = lb > 0.0f ? lb * sycl::tanh(up / lb) : up;
                    if (lane == 0) av[sgi * RV + r] = a * u;
                }
                sycl::group_barrier(it.get_group());
                if (sgi == 0) {
                    float* ap = av.get_multi_ptr<sycl::access::decorated::no>().get();
                    quant_block(sg, lane, ap, aq + p * F + ab * 32, ad + p * nblk + ab);
                }
            });
        });
    };
    if (type == 8) run(std::integral_constant<int, 8>{});
    else run(std::integral_constant<int, 40>{});
    return 0;
    NS_CATCH
}

int ns_ink2s_moe_down(ns_gpu* g, int type, const void* w, int64_t E, const float* scale, const int32_t* ids, const float* wts, const void* act_q,
                      float* y, int64_t T, int64_t K, int64_t F, int64_t n_out) {
    NS_TRY
    if (!ns_ink2s_supported(type, F) || K > 16 || K < 1) return ns_fail("ns_ink2s_moe_down: a type or shape it does not take");
    // a work-group a (row, 4 output rows): a sub-group an expert, its 4 rows; the K sums weighted in order
    constexpr int RD = 4;
    const int64_t P = T * K, nrb = (n_out + RD - 1) / RD, nl = type == 8 ? F / 16 : F / 32, nbx = F / 32;
    const int64_t rows = E * n_out, qr = qrow(type, F), sr = srow(type, F);
    const int8_t* wq = (const int8_t*) w;
    const uint8_t* ws = (const uint8_t*) w + ns_ink2s_scales_at(type, rows, F);
    const int8_t* aq = xq_q(act_q);
    const float* ad = xq_d(act_q, P, F);
    auto run = [&](auto tyc) {
        constexpr int TY = decltype(tyc)::value;
        g->q.submit([&](sycl::handler& h) {
            sycl::local_accessor<float, 1> part(sycl::range<1>(16 * RD), h);
            sycl::local_accessor<uint32_t, 1> lutm(sycl::range<1>(256), h);
            h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (T * nrb) * K * SG), sycl::range<1>((size_t) K * SG)),
                           [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                uint32_t* lut = lutm.get_multi_ptr<sycl::access::decorated::no>().get();
                if constexpr (TY == 40) fill_lut(it, lut);
                auto sg = it.get_sub_group();
                const int lane = (int) sg.get_local_linear_id(), k = (int) sg.get_group_linear_id();
                const int64_t t = it.get_group(0) / nrb, r0 = (it.get_group(0) % nrb) * RD;
                const int64_t p = t * K + k, e = ids[p];
                const int nr = (int) sycl::min<int64_t>(RD, n_out - r0);
                float acc[RD];
#pragma unroll
                for (int r = 0; r < RD; ++r) acc[r] = 0.0f;
                for (int64_t li = lane; li < nl; li += SG) {
                    const XLine<TY> x = load_x<TY>(aq + p * F, ad + p * nbx, li);
#pragma unroll
                    for (int r = 0; r < RD; ++r) {
                        const int64_t row = e * n_out + r0 + sycl::min(r, nr - 1);
                        acc[r] = ldot<TY>(acc[r], load_line<TY>(wq + row * qr, ws + row * sr, li, lut), x);
                    }
                }
                const float s = (scale ? scale[e] : 1.0f) * wts[p];
#pragma unroll
                for (int r = 0; r < RD; ++r) {
                    const float v = sycl::reduce_over_group(sg, acc[r], sycl::plus<float>());
                    if (lane == 0) part[k * RD + r] = v * s;
                }
                sycl::group_barrier(it.get_group());
                if (k == 0 && lane < nr) {
                    float v = 0.0f;
                    for (int kk = 0; kk < K; ++kk) v += part[kk * RD + lane];
                    y[t * n_out + r0 + lane] = v;
                }
            });
        });
    };
    if (type == 8) run(std::integral_constant<int, 8>{});
    else run(std::integral_constant<int, 40>{});
    return 0;
    NS_CATCH
}

}  // extern "C"
