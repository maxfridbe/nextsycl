// mm.cpp (the trellis2 silo): the flows' products with their epilogues fused, on oneDNN (the engine's mm.cpp, copied
// and changed). The engine's own path runs a product (the shared nsd_linear in 8192-row chunks, or its float32 one),
// then the GELU over the 8192-wide middle, then the gated residual add over the float32 stream, each a pass over
// memory: at 20k tokens the GELU and the adds took a fifth of a flow call's device time. Here they are the product's
// post-ops (eltwise GELU-tanh or SiLU before the rounding to half; binary_mul by the gate and sum into the residual), and a
// product is one primitive over all its rows. A primitive is made once a shape.
#include "ns.h"
#include "ns_internal.hpp"
#include "t2s.h"

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
    std::map<std::tuple<int64_t, int64_t, int64_t, int, int, int, int>, dnnl::matmul> prims;
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

int ns_t2s_linear_supported(int epi, int out_dt) {
    switch (epi) {
        case NS_T2S_PLAIN: case NS_T2S_GELU: case NS_T2S_SILU: return out_dt == 0 || out_dt == 1;
        case NS_T2S_RESID: return out_dt == 0;
        default: return 0;
    }
}

int ns_t2s_linear(ns_gpu* g, const void* x, int64_t M, int64_t K, const void* w, int64_t N, const float* bias, void* out, int out_dt, int epi,
                  const float* gate) {
    NS_TRY
    if (!ns_t2s_linear_supported(epi, out_dt)) return ns_fail("ns_t2s_linear: an epilogue or output type it does not take");
    if (M <= 0 || K <= 0 || N <= 0) return 0;
    std::lock_guard<std::mutex> l(mu);
    Dnn& d = dnn(g);
    const auto odt = out_dt == 0 ? memory::data_type::f32 : memory::data_type::f16;
    memory::desc smd({M, K}, memory::data_type::f16, memory::format_tag::ab);
    memory::desc wmd({K, N}, memory::data_type::f16, memory::format_tag::ba);   // the [N, K] buffer, read transposed
    memory::desc bmd({1, N}, memory::data_type::f32, memory::format_tag::ab);
    memory::desc dmd({M, N}, odt, memory::format_tag::ab);
    const int gated = epi == NS_T2S_RESID && gate ? 1 : 0;
    auto key = std::make_tuple(M, K, N, bias ? 1 : 0, out_dt, epi, gated);
    auto it = d.prims.find(key);
    if (it == d.prims.end()) {
        dnnl::post_ops po;
        if (epi == NS_T2S_GELU) po.append_eltwise(dnnl::algorithm::eltwise_gelu_tanh, 0.f, 0.f);
        if (epi == NS_T2S_SILU) po.append_eltwise(dnnl::algorithm::eltwise_swish, 1.f, 0.f);
        if (epi == NS_T2S_RESID) {
            if (gated) po.append_binary(dnnl::algorithm::binary_mul, bmd);
            po.append_sum(1.f);
        }
        dnnl::primitive_attr attr;
        attr.set_post_ops(po);
        auto pd = bias ? dnnl::matmul::primitive_desc(d.eng, smd, wmd, bmd, dmd, attr) : dnnl::matmul::primitive_desc(d.eng, smd, wmd, dmd, attr);
        it = d.prims.emplace(key, dnnl::matmul(pd)).first;
    }
    std::unordered_map<int, memory> args{{DNNL_ARG_SRC, usm(smd, d.eng, x)}, {DNNL_ARG_WEIGHTS, usm(wmd, d.eng, w)}, {DNNL_ARG_DST, usm(dmd, d.eng, out)}};
    if (bias) args.insert({DNNL_ARG_BIAS, usm(bmd, d.eng, bias)});
    if (gated) args.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, usm(bmd, d.eng, gate)});
    it->second.execute(d.st, args);
    return 0;
    NS_CATCH
}

}  // extern "C"
