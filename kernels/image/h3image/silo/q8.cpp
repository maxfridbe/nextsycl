// q8.cpp: the MiniMax-H3-Image silo's Q8_0 products (silo.h). The engine's k-quant path expands each block matrix to
// bfloat16 (2 bytes a value written, then read again by the product) before every product; a Q8_0 matrix is int8
// values with a half scale every 32 values along K, which is exactly what oneDNN's weight decompression reads (int8
// weights, grouped half scales, fpmath bf16 with integers included): the matrix is widened inside the product and the
// expansion pass is gone (HunyuanImage 2.1's silo does the same for its Q8_0 DiT). oneDNN wants the int8 values
// [N, K] and the scales [K / 32, N] apart, so each matrix is repacked once, at load, in its own buffer: 34 bytes a
// block of 32 either way.
#include "ns.h"
#include "ns_internal.hpp"
#include "silo.h"

#include <sycl/sycl.hpp>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>

namespace {
using dnnl::memory;
using dt = memory::data_type;

struct Dnn {
    dnnl::engine eng;
    dnnl::stream st;
    std::map<std::tuple<int64_t, int64_t, int64_t>, dnnl::matmul> prims;
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
}  // namespace

extern "C" {

int ns_image_h3i_silo_q8_repack(ns_gpu* g, void* blocks, int64_t N, int64_t K, void* tmp) try {
    if (N <= 0 || K <= 0 || K % 32 != 0) return ns_fail("h3image silo q8 repack: K a multiple of 32");
    const int64_t kb = K / 32;
    const uint8_t* src = static_cast<const uint8_t*>(blocks);
    int8_t* qv = static_cast<int8_t*>(tmp);
    uint16_t* sc = reinterpret_cast<uint16_t*>(qv + N * K);
    sycl::queue& q = g->q;
    // a work-item a block: its 32 values to row n, columns 32 b .., its scale to scales[b][n]
    q.parallel_for(sycl::range<2>((size_t) N, (size_t) kb), [=](sycl::id<2> id) {
        const int64_t n = id[0], b = id[1];
        const uint8_t* x = src + (n * kb + b) * 34;
        sc[b * N + n] = (uint16_t) (x[0] | (x[1] << 8));
        int8_t* o = qv + n * K + b * 32;
        for (int t = 0; t < 32; ++t) o[t] = (int8_t) x[2 + t];
    });
    q.memcpy(blocks, tmp, (size_t) (N * kb * 34));
    q.wait();
    return 0;
} catch (const std::exception& e) {
    return ns_fail(std::string("h3image silo q8 repack: ") + e.what());
}

int ns_image_h3i_silo_q8_linear(ns_gpu* g, const void* x, int64_t M, int64_t K, const void* w, int64_t N, void* out) try {
    if (M <= 0) return 0;
    if (K % 32 != 0) return ns_fail("h3image silo q8 linear: K a multiple of 32");
    std::lock_guard<std::mutex> lk(mu);
    Dnn& d = dnn(g);
    memory::desc src({M, K}, dt::bf16, memory::format_tag::ab);
    memory::desc wei({K, N}, dt::s8, memory::format_tag::ba);   // the [N, K] int8 values, read transposed
    memory::desc dst({M, N}, dt::bf16, memory::format_tag::ab);
    memory::desc smd({K / 32, N}, dt::f16, memory::format_tag::ab);
    auto key = std::make_tuple(M, K, N);
    auto it = d.prims.find(key);
    if (it == d.prims.end()) {
        if (d.prims.size() >= 256) {   // a new row count every request would grow it for as long as the server runs
            g->q.wait();
            d.prims.clear();
        }
        dnnl::primitive_attr attr;
        attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) | (1 << 1), {32, 1}, dt::f16);
        attr.set_fpmath_mode(dnnl::fpmath_mode::bf16, true);
        it = d.prims.emplace(key, dnnl::matmul(dnnl::matmul::primitive_desc(d.eng, src, wei, dst, attr))).first;
    }
    const int8_t* qv = static_cast<const int8_t*>(w);
    const void* sc = qv + N * K;
    it->second.execute(d.st, {{DNNL_ARG_SRC, usm(src, d.eng, x)}, {DNNL_ARG_WEIGHTS, usm(wei, d.eng, qv)}, {DNNL_ARG_DST, usm(dst, d.eng, out)},
                              {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, usm(smd, d.eng, sc)}});
    return 0;
} catch (const std::exception& e) {
    return ns_fail(std::string("h3image silo q8 linear: ") + e.what());
}

int ns_image_h3i_silo_linear(ns_gpu* g, const void* x, int64_t M, int64_t K, const void* w, int64_t N, void* out) try {
    if (M <= 0) return 0;
    std::lock_guard<std::mutex> lk(mu);
    Dnn& d = dnn(g);
    memory::desc src({M, K}, dt::bf16, memory::format_tag::ab);
    memory::desc wei({K, N}, dt::bf16, memory::format_tag::ba);   // the [N, K] matrix, read transposed
    memory::desc dst({M, N}, dt::bf16, memory::format_tag::ab);
    // keyed apart from the Q8_0 products: K negated
    auto key = std::make_tuple(M, -K, N);
    auto it = d.prims.find(key);
    if (it == d.prims.end()) {
        if (d.prims.size() >= 256) {
            g->q.wait();
            d.prims.clear();
        }
        it = d.prims.emplace(key, dnnl::matmul(dnnl::matmul::primitive_desc(d.eng, src, wei, dst))).first;
    }
    it->second.execute(d.st, {{DNNL_ARG_SRC, usm(src, d.eng, x)}, {DNNL_ARG_WEIGHTS, usm(wei, d.eng, w)}, {DNNL_ARG_DST, usm(dst, d.eng, out)}});
    return 0;
} catch (const std::exception& e) {
    return ns_fail(std::string("h3image silo linear: ") + e.what());
}

}  // extern "C"
