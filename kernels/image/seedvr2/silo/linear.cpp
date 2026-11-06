// linear.cpp: the VAE's pointwise (1x1x1) convolutions - the upsamplers' channel expansion and the resnets' shortcuts -
// as one oneDNN matmul over all the volume's pixels, bfloat16 in and out, the bias fused, the weights read as they are
// ([Co, Ci], transposed: no reordered copy). The engine's own path (the shared nsd_linear) goes 8,192 rows a call:
// at 864 x 480 a shortcut is 50 small products at 14 % of the card's peak (docs/ports/seedvr2.md). A primitive a
// shape, one user scratchpad.
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
#include <unordered_map>
#include <vector>

namespace {
using dnnl::memory;
using tag = memory::format_tag;
constexpr auto BF = memory::data_type::bf16;
constexpr size_t MAX_PRIMS = 128;

struct Prim {
    dnnl::matmul p;
    memory::desc scratch;
};

struct Dnn {
    dnnl::engine eng;
    dnnl::stream st;
    std::map<std::vector<int64_t>, Prim> prims;
    void* scratch = nullptr;
    size_t scratch_cap = 0;
};

std::mutex mu;
std::map<ns_gpu*, std::unique_ptr<Dnn>> ctxs;

Dnn& dnn(ns_gpu* g) {
    std::lock_guard<std::mutex> l(mu);
    auto& d = ctxs[g];
    if (!d) {
        d = std::make_unique<Dnn>();
        d->eng = dnnl::sycl_interop::make_engine(g->dev, g->ctx);
        d->st = dnnl::sycl_interop::make_stream(d->eng, g->q);
    }
    return *d;
}

memory mem(const memory::desc& md, const dnnl::engine& eng, const void* p) {
    return dnnl::sycl_interop::make_memory(md, eng, dnnl::sycl_interop::memory_kind::usm, const_cast<void*>(p));
}
}  // namespace

extern "C" {

int ns_image_seedvr2_silo_linear(ns_gpu* g, const void* x, int64_t M, int64_t K, const void* w, int64_t N, const float* bias, void* out) {
    NS_TRY
    if (M <= 0) return 0;
    Dnn& d = dnn(g);
    const memory::desc smd({M, K}, BF, tag::ab), wmd({K, N}, BF, tag::ba), dmd({M, N}, BF, tag::ab), bmd({1, N}, memory::data_type::f32, tag::ab);
    std::vector<int64_t> key{M, K, N, bias ? 1 : 0};
    auto it = d.prims.find(key);
    if (it == d.prims.end()) {
        dnnl::primitive_attr attr;
        attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
        auto pd = dnnl::matmul::primitive_desc(d.eng, smd, wmd, bias ? bmd : memory::desc(), dmd, attr);
        if (d.prims.size() >= MAX_PRIMS) {
            d.st.wait();
            d.prims.clear();
        }
        it = d.prims.emplace(key, Prim{dnnl::matmul(pd), pd.scratchpad_desc()}).first;
    }
    Prim& p = it->second;
    const size_t need = p.scratch.get_size();
    if (need > d.scratch_cap) {
        d.st.wait();
        if (d.scratch) sycl::free(d.scratch, g->q);
        d.scratch = sycl::malloc_device(need, g->q);
        d.scratch_cap = d.scratch ? need : 0;
        if (!d.scratch) return ns_fail("ns_image_seedvr2_silo_linear: no scratchpad of " + std::to_string(need >> 20) + " MiB");
    }
    std::unordered_map<int, memory> args{{DNNL_ARG_SRC, mem(smd, d.eng, x)}, {DNNL_ARG_WEIGHTS, mem(wmd, d.eng, w)}, {DNNL_ARG_DST, mem(dmd, d.eng, out)}};
    if (need) args.insert({DNNL_ARG_SCRATCHPAD, mem(p.scratch, d.eng, d.scratch)});
    if (bias) args.insert({DNNL_ARG_BIAS, mem(bmd, d.eng, bias)});
    p.p.execute(d.st, args);
    return 0;
    } catch (const dnnl::error& e) { return ns_fail(std::string("oneDNN (linear): ") + e.what());
    NS_CATCH
}

}  // extern "C"
