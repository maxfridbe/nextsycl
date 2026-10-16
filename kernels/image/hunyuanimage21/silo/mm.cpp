// mm.cpp: HunyuanImage 2.1's DiT products straight from its Q8_0 matrices (silo.h). The engine's own path expands
// each matrix to half (the shared dequant: ~1.2 ms for linear1's 90M values) and then multiplies on oneDNN; here
// oneDNN reads the matrix as int8 with a half scale every 32 values along K and widens it inside the product (its
// weight decompression: fpmath f16 with integers included), as fast as the half product itself on the B70 - the
// expansion pass is gone. The fusions the DiT wants come as oneDNN post-ops: GELU (tanh) on the MLP's first product,
// and the gated residual (x += gate * product) on the products that feed it. Inputs and outputs are read and written
// in place through strides (a column range of a wider row, a batch of guidance branches), so linear1 and linear2 of
// a single block need no concatenation pass.
#include "ns.h"
#include "ns_internal.hpp"
#include "silo.h"

#include <sycl/sycl.hpp>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>

#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace {
using dnnl::memory;
using dt = memory::data_type;
using half = sycl::half;

struct Dnn {
    dnnl::engine eng;
    dnnl::stream st;
    std::map<std::vector<int64_t>, dnnl::matmul> prims;
};

std::mutex mu;
std::map<ns_gpu*, std::unique_ptr<Dnn>> ctxs;

Dnn& dnn(ns_gpu* g) {
    auto& d = ctxs[g];
    if (!d) {
        d = std::make_unique<Dnn>();
        d->eng = dnnl::sycl_interop::make_engine(g->dev, g->ctx);
        d->st = dnnl::sycl_interop::make_stream(d->eng, g->q);
    }
    return *d;
}

memory usm(const memory::desc& md, const dnnl::engine& eng, const void* p) {
    return dnnl::sycl_interop::make_memory(md, eng, dnnl::sycl_interop::memory_kind::usm, const_cast<void*>(p));
}

// the descriptors of one call: 2-d for a single input, 3-d (the batch first; the matrix broadcast) for B > 1. With
// int8 (wscale) the input is the quantized copy, contiguous, and its row scales a column [M, 1] of their own.
struct Descs {
    memory::desc src, wei, dst, row, col, scales;
};

Descs descs(const ns_hi21_mm& a) {
    const dt odt = a.out_dt == 0 ? dt::f32 : dt::f16;
    const bool i8 = a.wscale != nullptr;
    Descs d;
    d.scales = memory::desc({a.kn / 32, a.N}, dt::f16, memory::format_tag::ab);
    if (a.B == 1) {
        d.src = i8 ? memory::desc({a.M, a.kn}, dt::s8, memory::format_tag::ab) : memory::desc({a.M, a.kn}, dt::f16, memory::dims{a.lda, 1});
        d.wei = memory::desc({a.kn, a.N}, dt::s8, memory::dims{1, a.K});
        d.dst = memory::desc({a.M, a.N}, odt, memory::dims{a.ldo, 1});
        d.col = memory::desc({1, a.N}, dt::f32, memory::format_tag::ab);
        d.row = memory::desc({a.M, 1}, dt::f32, memory::format_tag::ab);
    } else {
        d.src = i8 ? memory::desc({a.B, a.M, a.kn}, dt::s8, memory::format_tag::abc) : memory::desc({a.B, a.M, a.kn}, dt::f16, memory::dims{a.x_bs, a.lda, 1});
        d.wei = memory::desc({1, a.kn, a.N}, dt::s8, memory::dims{a.kn * a.K, 1, a.K});
        d.dst = memory::desc({a.B, a.M, a.N}, odt, memory::dims{a.o_bs, a.ldo, 1});
        d.col = memory::desc({1, 1, a.N}, dt::f32, memory::format_tag::abc);
        d.row = memory::desc({a.B, a.M, 1}, dt::f32, memory::format_tag::abc);
    }
    return d;
}

// the post-ops' order: int8 first undoes its scales (the row's, the column's) and adds the bias - all after the
// int32 product -, then the gate and the residual, or GELU
enum Arg { ROW, COL, BIAS, GATE };

std::vector<Arg> post_args(const ns_hi21_mm& a) {
    std::vector<Arg> v;
    if (a.wscale) {
        v.push_back(ROW);
        v.push_back(COL);
        if (a.bias) v.push_back(BIAS);
    }
    if (a.post == 2) v.push_back(GATE);
    return v;
}

dnnl::matmul prim(Dnn& d, const ns_hi21_mm& a, const Descs& ds) {
    const bool i8 = a.wscale != nullptr;
    const std::vector<int64_t> key{a.B, a.M, a.K, a.kn, a.N, a.lda, a.x_bs, a.ldo, a.o_bs, a.out_dt, a.post, a.bias ? 1 : 0, i8 ? 1 : 0};
    auto it = d.prims.find(key);
    if (it != d.prims.end()) return it->second;
    dnnl::primitive_attr attr;
    if (!i8) {
        // the scales: one a (32 values of K, column n) - mask over both of the matrix's own dimensions
        const int mask = a.B == 1 ? (1 << 0) | (1 << 1) : (1 << 1) | (1 << 2);
        attr.set_scales(DNNL_ARG_WEIGHTS, mask, {32, 1}, dt::f16);
        attr.set_fpmath_mode(dnnl::fpmath_mode::f16, true);
    }
    dnnl::post_ops po;
    for (Arg x : post_args(a)) {
        if (x == ROW) po.append_binary(dnnl::algorithm::binary_mul, ds.row);
        if (x == COL) po.append_binary(dnnl::algorithm::binary_mul, ds.col);
        if (x == BIAS) po.append_binary(dnnl::algorithm::binary_add, ds.col);
        if (x == GATE) {
            po.append_binary(dnnl::algorithm::binary_mul, ds.col);
            po.append_sum(1.0f);
        }
    }
    if (a.post == 1) po.append_eltwise(dnnl::algorithm::eltwise_gelu_tanh, 0.f, 0.f);
    attr.set_post_ops(po);
    auto pd = a.bias && !i8 ? dnnl::matmul::primitive_desc(d.eng, ds.src, ds.wei, ds.col, ds.dst, attr)
                            : dnnl::matmul::primitive_desc(d.eng, ds.src, ds.wei, ds.dst, attr);
    return d.prims.emplace(key, dnnl::matmul(pd)).first->second;
}

// The int8 path. Both factors are turned by the same orthonormal Hadamard matrix in blocks of 256 along K before they
// are quantized (x H . (W H)^T = x W^T): an outlier channel is spread over its block's 256, so one scale a row loses
// far less. A block lives in one sub-group, 8 values a lane (lane l holds values 8 l .. 8 l + 7): three butterfly
// stages in the lane's registers, five across lanes.
constexpr int SG = 32, QWG = 256, QV = 8, QPER = 8;
using h8 = sycl::vec<half, QV>;

inline void fwht256(const sycl::sub_group& sg, float (&v)[QV]) {
#pragma unroll
    for (int h = 1; h < QV; h <<= 1)
#pragma unroll
        for (int e = 0; e < QV; ++e)
            if (!(e & h)) {
                const float a = v[e], b = v[e + h];
                v[e] = a + b;
                v[e + h] = a - b;
            }
    const int lane = (int) sg.get_local_linear_id();
#pragma unroll
    for (int m = 1; m < SG; m <<= 1) {
        const bool hi = lane & m;
#pragma unroll
        for (int e = 0; e < QV; ++e) {
            const float o = sycl::permute_group_by_xor(sg, v[e], m);
            v[e] = hi ? o - v[e] : v[e] + o;
        }
    }
#pragma unroll
    for (int e = 0; e < QV; ++e) v[e] *= 0.0625f;   // 1 / sqrt(256)
}

// each row (of each batch) of x turned and quantized by its largest magnitude (scale = max / 127) into xq [B * M,
// kn] and xs [B * M]. A work-group a row, the row held in registers (kn at most 256 x 8 x 8); a sub-group's 32 lanes
// hold 256 consecutive values - one block.
void quantize(ns_gpu* g, const ns_hi21_mm& a) {
    const half* x = (const half*) a.x;
    int8_t* xq = a.xq;
    float* xs = a.xs;
    const int64_t M = a.M, kn = a.kn, lda = a.lda, x_bs = a.x_bs;
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (a.B * M * QWG)), sycl::range<1>(QWG)),
                      [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        const int64_t r = it.get_group(0), b = r / M, m = r % M;
        const int l = (int) it.get_local_id(0);
        const half* row = x + b * x_bs + m * lda;
        float v[QPER][QV];
        float mx = 0.0f;
#pragma unroll
        for (int j = 0; j < QPER; ++j) {
            const int64_t i = ((int64_t) j * QWG + l) * QV;
            // a block is all in or all out of the row (kn a multiple of 256): the sub-group stays whole
            if (((int64_t) j * QWG + (l & ~(SG - 1))) * QV < kn) {
                const h8 h = *(const h8*) (row + i);
#pragma unroll
                for (int e = 0; e < QV; ++e) v[j][e] = (float) h[e];
                fwht256(it.get_sub_group(), v[j]);
#pragma unroll
                for (int e = 0; e < QV; ++e) mx = sycl::fmax(mx, sycl::fabs(v[j][e]));
            }
        }
        mx = sycl::reduce_over_group(it.get_group(), mx, sycl::maximum<float>());
        const float scale = sycl::fmax(mx / 127.0f, 1e-30f);
        if (l == 0) xs[r] = scale;
#pragma unroll
        for (int j = 0; j < QPER; ++j) {
            const int64_t i = ((int64_t) j * QWG + l) * QV;
            if (i < kn) {
                sycl::vec<int8_t, QV> q;
#pragma unroll
                for (int e = 0; e < QV; ++e) q[e] = (int8_t) sycl::clamp(sycl::rint(v[j][e] / scale), -127.0f, 127.0f);
                *(sycl::vec<int8_t, QV>*) (xq + r * kn + i) = q;
            }
        }
    });
}

const char* bad(const ns_hi21_mm& a) {
    if (a.B < 1 || a.M < 0 || a.N <= 0 || a.kn <= 0) return "sizes";
    if (a.K % 32 || a.k0 % 32 || a.kn % 32 || a.k0 + a.kn > a.K) return "K, k0 and kn multiples of 32 inside the matrix";
    if (a.post < 0 || a.post > 2) return "post 0, 1 or 2";
    if (a.post == 1 && a.out_dt != 1) return "the GELU product writes half";
    if (a.post == 2 && (a.out_dt != 0 || !a.gate)) return "the gated residual is float32 with a gate";
    if (a.out_dt != 0 && a.out_dt != 1) return "out_dt 0 or 1";
    if (a.wscale && (!a.xq || !a.xs || a.kn > (int64_t) QWG * QV * QPER || a.kn % 256 || a.k0 % 256 || a.lda % QV || a.x_bs % QV))
        return "int8: scratch for the quantized input, k0 and kn multiples of 256, kn at most 16384, the rows 8-value aligned";
    return nullptr;
}
}  // namespace

extern "C" {

int ns_image_hi21_silo_q8_split(ns_gpu* g, const void* blocks, int64_t N, int64_t K, int8_t* w, void* s) {
    NS_TRY
    if (K % 32 != 0 || N <= 0) return ns_fail("ns_image_hi21_silo_q8_split: K a multiple of 32");
    const uint8_t* b = (const uint8_t*) blocks;
    half* sc = (half*) s;
    const int64_t nb = K / 32;
    // a work-item a block: its scale to the transposed table, its 32 values in place
    g->q.parallel_for(sycl::range<2>((size_t) N, (size_t) nb), [=](sycl::id<2> id) {
        const int64_t n = id[0], j = id[1];
        const uint8_t* x = b + (n * nb + j) * 34;
        half d;
        std::memcpy(&d, x, 2);
        sc[j * N + n] = d;
        int8_t* o = w + n * K + j * 32;
#pragma unroll
        for (int i = 0; i < 32; ++i) o[i] = (int8_t) x[2 + i];
    });
    return 0;
    NS_CATCH
}

int ns_image_hi21_silo_q8_rows(ns_gpu* g, const void* blocks, int64_t N, int64_t K, int8_t* w, float* scale) {
    NS_TRY
    if (K % 256 != 0 || N <= 0) return ns_fail("ns_image_hi21_silo_q8_rows: K a multiple of 256");
    const uint8_t* b = (const uint8_t*) blocks;
    const int64_t nb = K / 32, groups = K / 256;
    // a work-group a row, a sub-group a 256-block at a time (8 values a lane: a quarter of a Q8_0 block): the turned
    // row's largest magnitude, then the row again (turned again - cheaper than keeping it) over it
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (N * QWG)), sycl::range<1>(QWG)),
                      [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        const int64_t n = it.get_group(0);
        const auto sg = it.get_sub_group();
        const int64_t sgi = sg.get_group_linear_id(), nsg = QWG / SG, lane = sg.get_local_linear_id();
        const uint8_t* row = b + n * nb * 34;
        auto load = [&](int64_t gi, float (&v)[QV]) {
            const int64_t k = gi * 256 + lane * QV;          // 8 values inside one block
            const uint8_t* blk = row + (k / 32) * 34;
            half d;
            std::memcpy(&d, blk, 2);
#pragma unroll
            for (int e = 0; e < QV; ++e) v[e] = (float) d * (float) (int8_t) blk[2 + k % 32 + e];
            fwht256(sg, v);
        };
        float mx = 0.0f;
        for (int64_t gi = sgi; gi < groups; gi += nsg) {
            float v[QV];
            load(gi, v);
#pragma unroll
            for (int e = 0; e < QV; ++e) mx = sycl::fmax(mx, sycl::fabs(v[e]));
        }
        mx = sycl::reduce_over_group(it.get_group(), mx, sycl::maximum<float>());
        const float s = sycl::fmax(mx / 127.0f, 1e-30f);
        if (it.get_local_id(0) == 0) scale[n] = s;
        for (int64_t gi = sgi; gi < groups; gi += nsg) {
            float v[QV];
            load(gi, v);
            sycl::vec<int8_t, QV> q;
#pragma unroll
            for (int e = 0; e < QV; ++e) q[e] = (int8_t) sycl::clamp(sycl::rint(v[e] / s), -127.0f, 127.0f);
            *(sycl::vec<int8_t, QV>*) (w + n * K + gi * 256 + lane * QV) = q;
        }
    });
    return 0;
    NS_CATCH
}

int ns_image_hi21_silo_linear_supported(ns_gpu* g, int int8) {
    try {
        std::lock_guard<std::mutex> l(mu);
        Dnn& d = dnn(g);
        // each kind of call the engine makes, on small shapes: half and float32 outputs, GELU, the gated residual,
        // a column range, a batch
        for (int i = int8 ? 4 : 0; i < (int8 ? 6 : 4); ++i) {
            ns_hi21_mm a{};
            a.lda = 128; a.x_bs = 64 * 128; a.B = i == 3 ? 2 : 1; a.M = 64; a.K = 128; a.k0 = i == 2 ? 32 : 0; a.kn = i == 2 ? 96 : 128; a.N = 64;
            a.ldo = 64; a.o_bs = 64 * 64; a.out_dt = i == 0 || i == 1 ? 1 : 0; a.post = i == 1 ? 1 : i >= 2 ? 2 : 0;
            a.bias = (const float*) &a;            // only its presence counts here
            a.gate = (const float*) &a;
            if (i >= 4) {
                // int8: half out, and the gated residual
                a.K = a.kn = a.lda = 256;
                a.k0 = 0;
                a.x_bs = 64 * 256;
                a.wscale = (const float*) &a;
                a.xq = (int8_t*) &a;
                a.xs = (float*) &a;
                a.out_dt = i == 4 ? 1 : 0;
                a.post = i == 4 ? 0 : 2;
            }
            if (bad(a)) return 0;
            prim(d, a, descs(a));
        }
        return 1;
    } catch (...) {
        return 0;
    }
}

int ns_image_hi21_silo_linear(ns_gpu* g, const ns_hi21_mm* ap) {
    NS_TRY
    const ns_hi21_mm& a = *ap;
    if (const char* why = bad(a)) return ns_fail(std::string("ns_image_hi21_silo_linear: ") + why);
    if (a.M == 0) return 0;
    std::lock_guard<std::mutex> l(mu);
    Dnn& d = dnn(g);
    const Descs ds = descs(a);
    dnnl::matmul mm = prim(d, a, ds);
    std::unordered_map<int, memory> args{
        {DNNL_ARG_WEIGHTS, usm(ds.wei, d.eng, a.w + a.k0)},
        {DNNL_ARG_DST, usm(ds.dst, d.eng, a.out)},
    };
    if (a.wscale) {
        quantize(g, a);
        args.insert({DNNL_ARG_SRC, usm(ds.src, d.eng, a.xq)});
    } else {
        args.insert({DNNL_ARG_SRC, usm(ds.src, d.eng, a.x)});
        args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, usm(ds.scales, d.eng, (const half*) a.s + (a.k0 / 32) * a.N)});
        if (a.bias) args.insert({DNNL_ARG_BIAS, usm(ds.col, d.eng, a.bias)});
    }
    const std::vector<Arg> pa = post_args(a);
    for (size_t i = 0; i < pa.size(); ++i) {
        const int at = DNNL_ARG_ATTR_MULTIPLE_POST_OP((int) i) | DNNL_ARG_SRC_1;
        if (pa[i] == ROW) args.insert({at, usm(ds.row, d.eng, a.xs)});
        if (pa[i] == COL) args.insert({at, usm(ds.col, d.eng, a.wscale)});
        if (pa[i] == BIAS) args.insert({at, usm(ds.col, d.eng, a.bias)});
        if (pa[i] == GATE) args.insert({at, usm(ds.col, d.eng, a.gate)});
    }
    mm.execute(d.st, args);
    return 0;
    NS_CATCH
}

}  // extern "C"
