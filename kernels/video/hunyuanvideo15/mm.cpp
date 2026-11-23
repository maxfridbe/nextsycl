// mm.cpp: products with half inputs and a float32 output on oneDNN (hi21's mm.cpp, the strides added): the text
// encoder's products into its float32 residual, the VAE attention's scores. A primitive is made once a shape.
#include "ns.h"
#include "ns_internal.hpp"
#include "hv15.h"

#include <sycl/sycl.hpp>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>

#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace {
using dnnl::memory;

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
}  // namespace

extern "C" int ns_video_hv15_linear_f32(ns_gpu* g, const void* x, int64_t M, int64_t K, const void* w, int64_t ldw, int64_t N, const float* bias,
                                        float* out, int64_t ldo) {
    NS_TRY
    if (M <= 0 || K <= 0 || N <= 0) return 0;
    if (ldw == 0) ldw = K;
    if (ldo == 0) ldo = N;
    if (ldw < K || ldo < N) return ns_fail("ns_video_hv15_linear_f32: row strides shorter than the rows");
    std::lock_guard<std::mutex> l(mu);
    Dnn& d = dnn(g);
    memory::desc smd({M, K}, memory::data_type::f16, memory::format_tag::ab);
    memory::desc wmd({K, N}, memory::data_type::f16, memory::dims{1, ldw});   // the [N, K] rows, read transposed
    memory::desc bmd({1, N}, memory::data_type::f32, memory::format_tag::ab);
    memory::desc dmd({M, N}, memory::data_type::f32, memory::dims{ldo, 1});
    const std::vector<int64_t> key{M, K, N, ldw, ldo, bias ? 1 : 0};
    auto it = d.prims.find(key);
    if (it == d.prims.end()) {
        auto pd = bias ? dnnl::matmul::primitive_desc(d.eng, smd, wmd, bmd, dmd) : dnnl::matmul::primitive_desc(d.eng, smd, wmd, dmd);
        it = d.prims.emplace(key, dnnl::matmul(pd)).first;
    }
    std::unordered_map<int, memory> args{{DNNL_ARG_SRC, usm(smd, d.eng, x)}, {DNNL_ARG_WEIGHTS, usm(wmd, d.eng, w)}, {DNNL_ARG_DST, usm(dmd, d.eng, out)}};
    if (bias) args.insert({DNNL_ARG_BIAS, usm(bmd, d.eng, bias)});
    it->second.execute(d.st, args);
    return 0;
    NS_CATCH
}
