// mm.cpp: the TRELLIS.2 DiT's and decoders' products that feed a float32 residual stream, on oneDNN with half inputs and
// a float32 output (accumulated and written in float32; the shared nsd_linear writes its product in the inputs' type).
// A primitive is made once a shape. (The pattern of kernels/image/hunyuanimage21/mm.cpp, this engine's own copy.)
#include "ns.h"
#include "ns_internal.hpp"
#include "t2.h"

#include <sycl/sycl.hpp>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>

#include <map>
#include <memory>
#include <mutex>
#include <tuple>
#include <unordered_map>

namespace {
using dnnl::memory;

struct Dnn {
    dnnl::engine eng;
    dnnl::stream st;
    std::map<std::tuple<int64_t, int64_t, int64_t, int>, dnnl::matmul> prims;
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

extern "C" int ns_model3d_trellis2_linear_f32(ns_gpu* g, const void* x, int64_t M, int64_t K, const void* w, int64_t N, const float* bias, float* out) {
    NS_TRY
    if (M <= 0 || K <= 0 || N <= 0) return 0;
    std::lock_guard<std::mutex> l(mu);
    Dnn& d = dnn(g);
    memory::desc smd({M, K}, memory::data_type::f16, memory::format_tag::ab);
    memory::desc wmd({K, N}, memory::data_type::f16, memory::format_tag::ba);   // the [N, K] buffer, read transposed
    memory::desc bmd({1, N}, memory::data_type::f32, memory::format_tag::ab);
    memory::desc dmd({M, N}, memory::data_type::f32, memory::format_tag::ab);
    auto key = std::make_tuple(M, K, N, bias ? 1 : 0);
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
