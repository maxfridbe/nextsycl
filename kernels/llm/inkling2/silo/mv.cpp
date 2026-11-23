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

// ue4m3 by its bits: e >= 1 (1 + m / 8) 2^(e - 8) is float32 exponent e + 119 with m at the mantissa's top; e = 0
// m 2^-10; 0 and 0x7F (NaN) 0 - the same floats as ue4m3()
inline float ue4m3_bits(uint32_t x) {
    const uint32_t e = (x >> 3) & 0xFu, m = x & 7u;
    const float sub = (float) m * (1.0f / 1024.0f);
    const float nrm = sycl::bit_cast<float>(((e + 119u) << 23) | (m << 20));
    return (x == 0x7Fu) ? 0.0f : (e == 0u ? sub : nrm);
}

// four codes (a nibble in each byte's low half) to their signed byte values, in the word's arithmetic (no table):
// magnitude c = code & 7 is the value for c <= 4; for c >= 4 add (c & 3) + 2 (b0 b1) (4 6 8 12); then the sign by
// ones' complement plus one - no carry leaves a byte because the repacking stores -0 (code 8) as +0 (code 0)
inline uint32_t fp4_bytes(uint32_t t) {
    constexpr uint32_t ONE = 0x01010101u;
    const uint32_t c = t & 0x07070707u;
    const uint32_t ex = (t & 0x03030303u) + (((t << 1) & t) & 0x02020202u);
    const uint32_t hm = ((t >> 2) & ONE) * 0xFFu;
    const uint32_t m = c + (ex & hm);
    const uint32_t sg = (t >> 3) & ONE;
    return (m ^ (sg * 0xFFu)) + sg;
}

// four code bytes (c: bytes 0-3 of a sub-block) to their eight values as two words of signed bytes: lo the low nibbles'
// (values j), hi the high nibbles' (values j + 8)
inline void fp4_words(uint32_t c, const uint32_t* lut, int& lo, int& hi) {
    (void) lut;
    lo = (int) fp4_bytes(c & 0x0F0F0F0Fu);
    hi = (int) fp4_bytes((c >> 4) & 0x0F0F0F0Fu);
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

// a line as read: its 16 bytes and its scale bits (Q8_0: the half; NVFP4: the two UE4M3 bytes) - read first for all
// a sub-group's rows, so the loads are in flight together, then decoded
struct Raw {
    sycl::int4 v;
    uint16_t s;
};

template <int TY>
inline Raw load_raw(const int8_t* q, const uint8_t* s, int64_t li) {
    Raw r;
    r.v = *(const sycl::int4*) (q + li * 16);
    std::memcpy(&r.s, s + (TY == 8 ? (li >> 1) : li) * 2, 2);
    return r;
}

template <int TY>
inline Line<TY> decode_line(const Raw& r, const uint32_t* lut) {
    Line<TY> l;
    const sycl::int4 v = r.v;
    if constexpr (TY == 8) {
        l.w[0] = v.x(); l.w[1] = v.y(); l.w[2] = v.z(); l.w[3] = v.w();
        l.f[0] = (float) sycl::bit_cast<half>(r.s);
    } else {
        // the group's 16 bytes: sub-block 0's 8 then sub-block 1's; a byte's low nibble value j, its high one j + 8
        const uint32_t c0 = (uint32_t) v.x(), c1 = (uint32_t) v.y(), c2 = (uint32_t) v.z(), c3 = (uint32_t) v.w();
        fp4_words(c0, lut, l.w[0], l.w[2]);
        fp4_words(c1, lut, l.w[1], l.w[3]);
        fp4_words(c2, lut, l.w[4], l.w[6]);
        fp4_words(c3, lut, l.w[5], l.w[7]);
        l.f[0] = ue4m3_bits(r.s & 0xFFu);
        l.f[1] = ue4m3_bits((uint32_t) r.s >> 8);
    }
    return l;
}

template <int TY>
inline Line<TY> load_line(const int8_t* q, const uint8_t* s, int64_t li, const uint32_t* lut) {
    return decode_line<TY>(load_raw<TY>(q, s, li), lut);
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

// ---- dense products: R rows a sub-group (4, or 1 for launches of few rows: more sub-groups in flight; a row's sums
// are the same either way), NSG sub-groups a work-group ----
constexpr int NSG = 4;
constexpr int64_t FEW_ROWS = 16384;

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

template <int TY, int TT, int R>
void mv_launch(sycl::queue& q, Jobs jobs, int64_t blocks) {
    q.submit([&](sycl::handler& h) {
    h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) blocks * SG), sycl::range<1>(NSG * SG)), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        const uint32_t* lut = nullptr;  // (no table: the codes and scales decoded in arithmetic)
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
            Raw raw[R];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const int64_t row = r0 + sycl::min(r, nr - 1);
                raw[r] = load_raw<TY>(jb.q + row * qr, jb.s + row * sr, li);
            }
            Line<TY> w[R];
#pragma unroll
            for (int r = 0; r < R; ++r) w[r] = decode_line<TY>(raw[r], lut);
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

template <int TY, int R>
void mv_t(sycl::queue& q, const Jobs& jobs, int64_t blocks, int64_t T) {
    switch (T) {
        case 1: mv_launch<TY, 1, R>(q, jobs, blocks); break;
        case 2: mv_launch<TY, 2, R>(q, jobs, blocks); break;
        case 3: mv_launch<TY, 3, R>(q, jobs, blocks); break;
        case 4: mv_launch<TY, 4, R>(q, jobs, blocks); break;
        case 5: mv_launch<TY, 5, R>(q, jobs, blocks); break;
        case 6: mv_launch<TY, 6, R>(q, jobs, blocks); break;
        case 7: mv_launch<TY, 7, R>(q, jobs, blocks); break;
        case 8: mv_launch<TY, 8, R>(q, jobs, blocks); break;
        case 9: mv_launch<TY, 9, R>(q, jobs, blocks); break;
        case 10: mv_launch<TY, 10, R>(q, jobs, blocks); break;
        case 11: mv_launch<TY, 11, R>(q, jobs, blocks); break;
        case 12: mv_launch<TY, 12, R>(q, jobs, blocks); break;
        case 13: mv_launch<TY, 13, R>(q, jobs, blocks); break;
        case 14: mv_launch<TY, 14, R>(q, jobs, blocks); break;
        case 15: mv_launch<TY, 15, R>(q, jobs, blocks); break;
        default: mv_launch<TY, 16, R>(q, jobs, blocks); break;
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
            // -0 (code 8) stored as +0 (code 0): the same value, and the decode's sign step never carries
            for (int i = 0; i < 32; ++i) {
                uint8_t c = p[4 + i];
                if ((c & 0x0F) == 0x08) c &= 0xF0;
                if ((c & 0xF0) == 0x80) c &= 0x0F;
                q[b * 32 + i] = (int8_t) c;
            }
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
    // a work-item 16 values, written as two 16-byte stores
    auto put = [](uint16_t* d, const uint16_t* v) {
        sycl::uint4 lo, hi;
        lo.x() = v[0] | ((uint32_t) v[1] << 16); lo.y() = v[2] | ((uint32_t) v[3] << 16);
        lo.z() = v[4] | ((uint32_t) v[5] << 16); lo.w() = v[6] | ((uint32_t) v[7] << 16);
        hi.x() = v[8] | ((uint32_t) v[9] << 16); hi.y() = v[10] | ((uint32_t) v[11] << 16);
        hi.z() = v[12] | ((uint32_t) v[13] << 16); hi.w() = v[14] | ((uint32_t) v[15] << 16);
        ((sycl::uint4*) d)[0] = lo;
        ((sycl::uint4*) d)[1] = hi;
    };
    if (type == 8) {
        g->q.parallel_for(sycl::range<2>((size_t) n, (size_t) per), [=](sycl::id<2> id) {
            const int64_t r = id[0], c = id[1], row = row0 + r;
            uint16_t h;
            std::memcpy(&h, s + row * sr + (c >> 1) * 2, 2);
            const float d = (float) sycl::bit_cast<half>(h);
            const sycl::vec<int8_t, 16> qv = *(const sycl::vec<int8_t, 16>*) (q + row * qr + c * 16);
            uint16_t v[16];
#pragma unroll
            for (int i = 0; i < 16; ++i) v[i] = sycl::bit_cast<uint16_t>(half(d * (float) qv[i]));
            put(dst + r * n_in + c * 16, v);
        });
    } else {
        g->q.parallel_for(sycl::range<2>((size_t) n, (size_t) per), [=](sycl::id<2> id) {
            const int64_t r = id[0], c = id[1], row = row0 + r;
            // sub-block c: its 8 code bytes, its scale
            const float d = ue4m3(s[row * sr + c]);
            const sycl::vec<uint8_t, 8> cb = *(const sycl::vec<uint8_t, 8>*) ((const uint8_t*) q + row * qr + c * 8);
            constexpr float t[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
            uint16_t v[16];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                v[j] = sycl::bit_cast<uint16_t>(half(t[cb[j] & 15] * d));
                v[j + 8] = sycl::bit_cast<uint16_t>(half(t[cb[j] >> 4] * d));
            }
            put(dst + r * n_in + c * 16, v);
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

int ns_ink2s_rms_xq_many(ns_gpu* g, int m, const float* const* xs, const float* const* ws, float* const* ys, void* const* xqs, int64_t T, int64_t n,
                         float eps) {
    NS_TRY
    if (n % 32 || n > 4096 || m < 1 || m > 4) return ns_fail("ns_ink2s_rms_xq_many: 1..4 inputs, rows of whole blocks of 32, at most 4096");
    constexpr int W = 256;
    struct Jb {
        const float* x[4];
        const float* w[4];
        float* y[4];
        int8_t* q[4];
        float* d[4];
    } jb{};
    for (int k = 0; k < m; ++k) {
        jb.x[k] = xs[k];
        jb.w[k] = ws[k];
        jb.y[k] = ys ? ys[k] : nullptr;
        jb.q[k] = (int8_t*) xqs[k];
        jb.d[k] = (float*) ((uint8_t*) xqs[k] + ns_ink2s_xq_scales_at(T, n));
    }
    if (T > 0) g->q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> row(sycl::range<1>(4096), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (m * T) * W), sycl::range<1>(W)), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const int64_t t = it.get_group(0) % T;
            const int k = (int) (it.get_group(0) / T);
            const int tid = (int) it.get_local_id(0);
            const float* xr = jb.x[k] + t * n;
            const float* w = jb.w[k];
            float* y = jb.y[k];
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
            for (int64_t b = sgi; b < n / 32; b += W / SG) quant_block(sg, lane, rp + b * 32, jb.q[k] + t * n + b * 32, jb.d[k] + t * (n / 32) + b);
        });
    });
    return 0;
    NS_CATCH
}

int ns_ink2s_rms_xq(ns_gpu* g, const float* x, const float* w, float* y, void* xq, int64_t T, int64_t n, float eps) {
    return ns_ink2s_rms_xq_many(g, 1, &x, &w, y ? &y : nullptr, &xq, T, n, eps);
}

int ns_ink2s_mv_many(ns_gpu* g, int n, const int* ty, const void* const* w, const void* const* xq, float* const* y, const int64_t* rows,
                     const int64_t* n_in, int64_t T) {
    NS_TRY
    if (n < 1 || n > 8 || T < 1 || T > MAXT) return ns_fail("ns_ink2s_mv_many: 1..8 products, 1..16 rows");
    // the jobs of one type a launch (the launches in order); a launch of few rows takes a row a sub-group
    for (int i = 0; i < n;) {
        const int t0 = ty[i];
        int k = i;
        int64_t total = 0;
        for (; k < n && ty[k] == t0 && k - i < 8; ++k) total += rows[k];
        const int R = total < FEW_ROWS ? 1 : 4;
        Jobs js{};
        int64_t blocks = 0;
        for (; i < k; ++i) {
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
        if (t0 == 8) {
            if (R == 1) mv_t<8, 1>(g->q, js, wgs * NSG, T);
            else mv_t<8, 4>(g->q, js, wgs * NSG, T);
        } else {
            if (R == 1) mv_t<40, 1>(g->q, js, wgs * NSG, T);
            else mv_t<40, 4>(g->q, js, wgs * NSG, T);
        }
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
    constexpr int RV = 4, NS8 = 32 / RV;
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
            h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (P * nblk) * NS8 * SG), sycl::range<1>(NS8 * SG)),
                           [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                const uint32_t* lut = nullptr;  // (no table: the codes and scales decoded in arithmetic)
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
            h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (T * nrb) * K * SG), sycl::range<1>((size_t) K * SG)),
                           [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                const uint32_t* lut = nullptr;  // (no table: the codes and scales decoded in arithmetic)
                auto sg = it.get_sub_group();
                const int lane = (int) sg.get_local_linear_id(), k = (int) sg.get_group_linear_id();
                const int64_t t = it.get_group(0) / nrb, r0 = (it.get_group(0) % nrb) * RD;
                const int64_t p = t * K + k, e = ids[p];
                const int nr = (int) sycl::min<int64_t>(RD, n_out - r0);
                float acc[RD];
#pragma unroll
                for (int r = 0; r < RD; ++r) acc[r] = 0.0f;
                for (int64_t li = lane; li < nl; li += SG) {
                    Raw raw[RD];
#pragma unroll
                    for (int r = 0; r < RD; ++r) {
                        const int64_t row = e * n_out + r0 + sycl::min(r, nr - 1);
                        raw[r] = load_raw<TY>(wq + row * qr, ws + row * sr, li);
                    }
                    const XLine<TY> x = load_x<TY>(aq + p * F, ad + p * nbx, li);
#pragma unroll
                    for (int r = 0; r < RD; ++r) acc[r] = ldot<TY>(acc[r], decode_line<TY>(raw[r], lut), x);
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

int ns_ink2s_moe_union(ns_gpu* g, const int32_t* ids, int64_t P, int32_t* uni) {
    NS_TRY
    if (P < 1 || P > 256) return ns_fail("ns_ink2s_moe_union: 1..256 pairs");
    // uni: [0] the distinct experts' count n, then their experts [P] (in order of first use), each one's pair count [P],
    // its pairs [P][16] (in pair order); a work-item a pair
    g->q.submit([&](sycl::handler& h) {
        sycl::local_accessor<int32_t, 1> fp(sycl::range<1>(256), h);
        sycl::local_accessor<int32_t, 1> isf(sycl::range<1>(256), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(256), sycl::range<1>(256)), [=](sycl::nd_item<1> it) {
            const int p = (int) it.get_local_id(0);
            int32_t* ue = uni + 1;
            int32_t* un = ue + P;
            int32_t* up = un + P;
            int f = p, slot = 0;
            if (p < P) {
                const int32_t e = ids[p];
                for (int q = p - 1; q >= 0; --q)
                    if (ids[q] == e) {
                        f = q;
                        ++slot;
                    }
            }
            fp[p] = f;
            isf[p] = p < P && f == p;
            sycl::group_barrier(it.get_group());
            if (p < P) {
                // the expert's distinct index: the first uses before its first pair
                int u = 0;
                for (int q = 0; q < f; ++q) u += isf[q];
                up[u * 16 + slot] = p;
                if (f == p) ue[u] = ids[p];
                bool last = true;
                for (int q = p + 1; q < P; ++q)
                    if (fp[q] == f) {
                        last = false;
                        break;
                    }
                if (last) un[u] = slot + 1;
                if (p == 0) {
                    int c = 0;
                    for (int q = 0; q < P; ++q) c += isf[q];
                    uni[0] = c;
                }
            }
        });
    });
    return 0;
    NS_CATCH
}

int ns_ink2s_moe_gu_union(ns_gpu* g, int type, const void* w, int64_t E, const float* scale, const int32_t* uni, const void* xq, void* act_q, int64_t T,
                          int64_t K, int64_t n_in, int64_t F, float beta, float lb) {
    NS_TRY
    if (!ns_ink2s_supported(type, n_in) || F % 32 || T * K > 256 || T > 16) return ns_fail("ns_ink2s_moe_gu_union: a type or shape it does not take");
    // a work-group a (distinct expert, 32 act values): 8 sub-groups of 4 values, each 4 gate and 4 up rows, its lines
    // read once for every pair of the expert (up to 4 at a time) - each pair's sums as ns_ink2s_moe_gu's
    constexpr int NS8 = 8, RV = 4, MP = 4;
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
            sycl::local_accessor<float, 1> av(sycl::range<1>(MP * 32), h);
            h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (P * nblk) * NS8 * SG), sycl::range<1>(NS8 * SG)),
                           [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                const int64_t u = it.get_group(0) / nblk, ab = it.get_group(0) % nblk;
                if (u >= uni[0]) return;
                const uint32_t* lut = nullptr;  // (no table: the codes and scales decoded in arithmetic)
                auto sg = it.get_sub_group();
                const int lane = (int) sg.get_local_linear_id(), sgi = (int) sg.get_group_linear_id();
                const int32_t* ue = uni + 1;
                const int32_t* un = ue + P;
                const int32_t* up = un + P;
                const int64_t e = ue[u];
                const int m = un[u];
                const int64_t base = e * 2 * F, i0 = ab * 32 + sgi * RV;
                const float s = scale ? scale[e] : 1.0f;
                for (int g0 = 0; g0 < m; g0 += MP) {
                    const int mg = sycl::min(MP, m - g0);
                    int64_t pt[MP];
#pragma unroll
                    for (int j = 0; j < MP; ++j) pt[j] = up[u * 16 + g0 + sycl::min(j, mg - 1)] / K;
                    float acc[MP][2 * RV];
#pragma unroll
                    for (int j = 0; j < MP; ++j)
#pragma unroll
                        for (int r = 0; r < 2 * RV; ++r) acc[j][r] = 0.0f;
                    for (int64_t li = lane; li < nl; li += SG) {
                        XLine<TY> x[MP];
#pragma unroll
                        for (int j = 0; j < MP; ++j)
                            if (j < mg) x[j] = load_x<TY>(xqq + pt[j] * n_in, xqd + pt[j] * nbx, li);
#pragma unroll
                        for (int r = 0; r < 2 * RV; ++r) {
                            const int64_t row = base + (r < RV ? i0 + r : F + i0 + (r - RV));
                            const Line<TY> wl = load_line<TY>(wq + row * qr, ws + row * sr, li, lut);
#pragma unroll
                            for (int j = 0; j < MP; ++j)
                                if (j < mg) acc[j][r] = ldot<TY>(acc[j][r], wl, x[j]);
                        }
                    }
#pragma unroll
                    for (int j = 0; j < MP; ++j) {
                        if (j >= mg) break;
#pragma unroll
                        for (int r = 0; r < RV; ++r) {
                            const float gt = sycl::reduce_over_group(sg, acc[j][r], sycl::plus<float>()) * s;
                            const float upv = sycl::reduce_over_group(sg, acc[j][RV + r], sycl::plus<float>()) * s;
                            const float a = beta * sycl::tanh(gt / beta) * (1.0f / (1.0f + sycl::exp(-gt)));
                            const float uu = lb > 0.0f ? lb * sycl::tanh(upv / lb) : upv;
                            if (lane == 0) av[j * 32 + sgi * RV + r] = a * uu;
                        }
                    }
                    sycl::group_barrier(it.get_group());
                    if (sgi < mg) {
                        const int64_t p = up[u * 16 + g0 + sgi];
                        float* ap = av.get_multi_ptr<sycl::access::decorated::no>().get() + sgi * 32;
                        quant_block(sg, lane, ap, aq + p * F + ab * 32, ad + p * nblk + ab);
                    }
                    sycl::group_barrier(it.get_group());
                }
            });
        });
    };
    if (type == 8) run(std::integral_constant<int, 8>{});
    else run(std::integral_constant<int, 40>{});
    return 0;
    NS_CATCH
}

int ns_ink2s_moe_down_union(ns_gpu* g, int type, const void* w, int64_t E, const float* scale, const int32_t* uni, const float* wts, const void* act_q,
                            float* part, float* y, int64_t T, int64_t K, int64_t F, int64_t n_out) {
    NS_TRY
    if (!ns_ink2s_supported(type, F) || K > 16 || K < 1 || T * K > 256) return ns_fail("ns_ink2s_moe_down_union: a type or shape it does not take");
    // a work-group a (distinct expert, 16 output rows): a sub-group 4 rows, its lines read once for every pair of the
    // expert (up to 4 at a time), each pair's weighted sums into part [P, n_out]; then each row's K sums in order
    constexpr int RD = 4, NSG = 4, MP = 4;
    const int64_t P = T * K, nrb = (n_out + RD * NSG - 1) / (RD * NSG), nl = type == 8 ? F / 16 : F / 32, nbx = F / 32;
    const int64_t rows = E * n_out, qr = qrow(type, F), sr = srow(type, F);
    const int8_t* wq = (const int8_t*) w;
    const uint8_t* ws = (const uint8_t*) w + ns_ink2s_scales_at(type, rows, F);
    const int8_t* aq = xq_q(act_q);
    const float* ad = xq_d(act_q, P, F);
    auto run = [&](auto tyc) {
        constexpr int TY = decltype(tyc)::value;
        g->q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (P * nrb) * NSG * SG), sycl::range<1>(NSG * SG)),
                           [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                const int64_t u = it.get_group(0) / nrb, rb = it.get_group(0) % nrb;
                if (u >= uni[0]) return;
                const uint32_t* lut = nullptr;  // (no table: the codes and scales decoded in arithmetic)
                auto sg = it.get_sub_group();
                const int lane = (int) sg.get_local_linear_id(), sgi = (int) sg.get_group_linear_id();
                const int32_t* ue = uni + 1;
                const int32_t* un = ue + P;
                const int32_t* up = un + P;
                const int64_t e = ue[u];
                const int m = un[u];
                const int64_t r0 = (rb * NSG + sgi) * RD;
                if (r0 >= n_out) return;
                const int nr = (int) sycl::min<int64_t>(RD, n_out - r0);
                const float s = scale ? scale[e] : 1.0f;
                for (int g0 = 0; g0 < m; g0 += MP) {
                    const int mg = sycl::min(MP, m - g0);
                    int64_t pp[MP];
#pragma unroll
                    for (int j = 0; j < MP; ++j) pp[j] = up[u * 16 + g0 + sycl::min(j, mg - 1)];
                    float acc[MP][RD];
#pragma unroll
                    for (int j = 0; j < MP; ++j)
#pragma unroll
                        for (int r = 0; r < RD; ++r) acc[j][r] = 0.0f;
                    for (int64_t li = lane; li < nl; li += SG) {
                        Raw raw[RD];
#pragma unroll
                        for (int r = 0; r < RD; ++r) {
                            const int64_t row = e * n_out + r0 + sycl::min(r, nr - 1);
                            raw[r] = load_raw<TY>(wq + row * qr, ws + row * sr, li);
                        }
                        XLine<TY> x[MP];
#pragma unroll
                        for (int j = 0; j < MP; ++j)
                            if (j < mg) x[j] = load_x<TY>(aq + pp[j] * F, ad + pp[j] * nbx, li);
#pragma unroll
                        for (int r = 0; r < RD; ++r) {
                            const Line<TY> wl = decode_line<TY>(raw[r], lut);
#pragma unroll
                            for (int j = 0; j < MP; ++j)
                                if (j < mg) acc[j][r] = ldot<TY>(acc[j][r], wl, x[j]);
                        }
                    }
#pragma unroll
                    for (int j = 0; j < MP; ++j) {
                        if (j >= mg) break;
                        const float sw = s * wts[pp[j]];
#pragma unroll
                        for (int r = 0; r < RD; ++r) {
                            const float v = sycl::reduce_over_group(sg, acc[j][r], sycl::plus<float>());
                            if (lane == 0 && r < nr) part[pp[j] * n_out + r0 + r] = v * sw;
                        }
                    }
                }
            });
        });
        // each row's sums over its K experts, in order (ns_ink2s_moe_down's)
        g->q.parallel_for(sycl::range<1>((size_t) (T * n_out)), [=](sycl::id<1> id) {
            const int64_t t = id[0] / n_out, r = id[0] % n_out;
            float v = 0.0f;
            for (int64_t k = 0; k < K; ++k) v += part[(t * K + k) * n_out + r];
            y[t * n_out + r] = v;
        });
    };
    if (type == 8) run(std::integral_constant<int, 8>{});
    else run(std::integral_constant<int, 40>{});
    return 0;
    NS_CATCH
}

}  // extern "C"
