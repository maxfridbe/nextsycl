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

// the descriptors of one call: 2-d for a single input, 3-d (the batch first; the matrix broadcast) for B > 1
struct Descs {
    memory::desc src, wei, dst, bias, gate, scales;
};

Descs descs(const ns_hi21_mm& a) {
    const dt odt = a.out_dt == 0 ? dt::f32 : dt::f16;
    Descs d;
    d.scales = memory::desc({a.kn / 32, a.N}, dt::f16, memory::format_tag::ab);
    if (a.B == 1) {
        d.src = memory::desc({a.M, a.kn}, dt::f16, memory::dims{a.lda, 1});
        d.wei = memory::desc({a.kn, a.N}, dt::s8, memory::dims{1, a.K});
        d.dst = memory::desc({a.M, a.N}, odt, memory::dims{a.ldo, 1});
        d.bias = memory::desc({1, a.N}, dt::f32, memory::format_tag::ab);
        d.gate = d.bias;
    } else {
        d.src = memory::desc({a.B, a.M, a.kn}, dt::f16, memory::dims{a.x_bs, a.lda, 1});
        d.wei = memory::desc({1, a.kn, a.N}, dt::s8, memory::dims{a.kn * a.K, 1, a.K});
        d.dst = memory::desc({a.B, a.M, a.N}, odt, memory::dims{a.o_bs, a.ldo, 1});
        d.bias = memory::desc({1, 1, a.N}, dt::f32, memory::format_tag::abc);
        d.gate = d.bias;
    }
    return d;
}

dnnl::matmul prim(Dnn& d, const ns_hi21_mm& a, const Descs& ds) {
    const std::vector<int64_t> key{a.B, a.M, a.K, a.kn, a.N, a.lda, a.x_bs, a.ldo, a.o_bs, a.out_dt, a.post, a.bias ? 1 : 0};
    auto it = d.prims.find(key);
    if (it != d.prims.end()) return it->second;
    dnnl::primitive_attr attr;
    // the scales: one a (32 values of K, column n) - mask over both of the matrix's own dimensions
    const int mask = a.B == 1 ? (1 << 0) | (1 << 1) : (1 << 1) | (1 << 2);
    attr.set_scales(DNNL_ARG_WEIGHTS, mask, {32, 1}, dt::f16);
    attr.set_fpmath_mode(dnnl::fpmath_mode::f16, true);
    dnnl::post_ops po;
    if (a.post == 1) po.append_eltwise(dnnl::algorithm::eltwise_gelu_tanh, 0.f, 0.f);
    if (a.post == 2) {
        po.append_binary(dnnl::algorithm::binary_mul, ds.gate);
        po.append_sum(1.0f);
    }
    attr.set_post_ops(po);
    auto pd = a.bias ? dnnl::matmul::primitive_desc(d.eng, ds.src, ds.wei, ds.bias, ds.dst, attr)
                     : dnnl::matmul::primitive_desc(d.eng, ds.src, ds.wei, ds.dst, attr);
    return d.prims.emplace(key, dnnl::matmul(pd)).first->second;
}

const char* bad(const ns_hi21_mm& a) {
    if (a.B < 1 || a.M < 0 || a.N <= 0 || a.kn <= 0) return "sizes";
    if (a.K % 32 || a.k0 % 32 || a.kn % 32 || a.k0 + a.kn > a.K) return "K, k0 and kn multiples of 32 inside the matrix";
    if (a.post < 0 || a.post > 2) return "post 0, 1 or 2";
    if (a.post == 1 && a.out_dt != 1) return "the GELU product writes half";
    if (a.post == 2 && (a.out_dt != 0 || !a.gate)) return "the gated residual is float32 with a gate";
    if (a.out_dt != 0 && a.out_dt != 1) return "out_dt 0 or 1";
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

int ns_image_hi21_silo_linear_supported(ns_gpu* g) {
    try {
        std::lock_guard<std::mutex> l(mu);
        Dnn& d = dnn(g);
        // each kind of call the engine makes, on small shapes: half and float32 outputs, GELU, the gated residual,
        // a column range, a batch
        for (int i = 0; i < 4; ++i) {
            ns_hi21_mm a{};
            a.lda = 128; a.x_bs = 64 * 128; a.B = i == 3 ? 2 : 1; a.M = 64; a.K = 128; a.k0 = i == 2 ? 32 : 0; a.kn = i == 2 ? 96 : 128; a.N = 64;
            a.ldo = 64; a.o_bs = 64 * 64; a.out_dt = i == 0 || i == 1 ? 1 : 0; a.post = i == 1 ? 1 : i >= 2 ? 2 : 0;
            a.bias = (const float*) &a;            // only its presence counts here
            a.gate = (const float*) &a;
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
        {DNNL_ARG_SRC, usm(ds.src, d.eng, a.x)},
        {DNNL_ARG_WEIGHTS, usm(ds.wei, d.eng, a.w + a.k0)},
        {DNNL_ARG_DST, usm(ds.dst, d.eng, a.out)},
        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, usm(ds.scales, d.eng, (const half*) a.s + (a.k0 / 32) * a.N)},
    };
    if (a.bias) args.insert({DNNL_ARG_BIAS, usm(ds.bias, d.eng, a.bias)});
    if (a.post == 2) args.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, usm(ds.gate, d.eng, a.gate)});
    mm.execute(d.st, args);
    return 0;
    NS_CATCH
}

}  // extern "C"
