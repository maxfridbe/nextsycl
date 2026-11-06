// conv.cpp: the upscalers' convolutions on oneDNN (implicit GEMM: no im2col copy), the zero padding taken by the
// primitive itself (no padded copy of the input) and the activation fused as a post-op (no pass over the output).
// A primitive a shape (cached), each weight reordered once into the primitive's layout (keyed by its pointer and the
// layout), one user scratchpad a GPU.
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
#include <string>
#include <unordered_map>
#include <vector>

namespace {
using dnnl::memory;
using tag = memory::format_tag;
constexpr auto F16 = memory::data_type::f16;
constexpr auto F32 = memory::data_type::f32;
constexpr size_t MAX_PRIMS = 512;

struct Prim {
    dnnl::convolution_forward p;
    memory::desc src, wei, dst, scratch;
    int prelu_at, add_at;   // the post-ops' places (-1: none)
};

struct Dnn {
    dnnl::engine eng;
    dnnl::stream st;
    std::map<std::vector<int64_t>, Prim> prims;
    // a weight's reordered copies, one a layout
    std::map<const void*, std::vector<memory>> weights;
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

int ns_image_up_silo_supported(ns_gpu* g) {
    return g ? 1 : 0;
}

int ns_image_up_silo_conv(ns_gpu* g, const void* x, int64_t H, int64_t W, int64_t Ci, int64_t x_ld, const void* w, int64_t Co, int64_t k,
                          int64_t stride, int64_t pt, int64_t pl, int64_t pb, int64_t pr, const float* bias, int act, float a, const float* slopes,
                          int64_t n_slopes, const void* res, float ra, float rb, void* out, int64_t out_ld) {
    NS_TRY
    if (stride < 1 || k < 1) return ns_fail("ns_image_up_silo_conv: a kernel and stride of at least 1");
    const int64_t Ho = (H + pt + pb - k) / stride + 1, Wo = (W + pl + pr - k) / stride + 1;
    if (Ho <= 0 || Wo <= 0) return ns_fail("ns_image_up_silo_conv: the picture is smaller than the kernel");
    if (act == 3 && !(slopes && (n_slopes == 1 || n_slopes == Co))) return ns_fail("ns_image_up_silo_conv: PReLU needs 1 or Co slopes");
    Dnn& d = dnn(g);
    if (x_ld == 0) x_ld = Ci;
    if (out_ld == 0) out_ld = Co;
    if (x_ld < Ci || out_ld < Co) return ns_fail("ns_image_up_silo_conv: a pixel's stride shorter than its channels");
    // channels-last, a pixel x_ld / out_ld halves apart: a channel slice of a wider buffer (a dense block's
    // concatenation, read or written in place)
    const memory::desc smd({1, Ci, H, W}, F16, memory::dims{H * W * x_ld, 1, W * x_ld, x_ld});
    const memory::desc dmd({1, Co, Ho, Wo}, F16, memory::dims{Ho * Wo * out_ld, 1, Wo * out_ld, out_ld});
    const memory::desc pmd = act == 3 ? memory::desc(n_slopes == 1 ? memory::dims{1, 1, 1, 1} : memory::dims{1, Co, 1, 1}, F32, tag::abcd)
                                      : memory::desc();
    std::vector<int64_t> key{H, W, Ci, x_ld, Co, out_ld, k, stride, pt, pl, pb, pr, bias ? 1 : 0, act, act == 3 ? n_slopes : 0, res ? 1 : 0};
    for (float f : {a, ra, rb}) {
        int32_t bits;
        std::memcpy(&bits, &f, 4);
        key.push_back(bits);
    }
    // the residual (dense, Co channels) added after the activation: ra x conv + rb x res = rb (ra / rb conv + res)
    const memory::desc rmd({1, Co, Ho, Wo}, F16, tag::nhwc);
    int prelu_at = -1, add_at = -1;
    auto it = d.prims.find(key);
    if (it == d.prims.end()) {
        dnnl::primitive_attr attr;
        attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
        dnnl::post_ops po;
        switch (act) {
            case 0: break;
            case 1: po.append_eltwise(dnnl::algorithm::eltwise_relu, 0.0f, 0.0f); break;
            case 2: po.append_eltwise(dnnl::algorithm::eltwise_relu, a, 0.0f); break;
            case 3: prelu_at = po.len(); po.append_prelu(n_slopes == 1 ? 0 : 1 << 1); break;
            case 4: po.append_eltwise(dnnl::algorithm::eltwise_logistic, 0.0f, 0.0f); break;
            default: return ns_fail("ns_image_up_silo_conv: activation " + std::to_string(act) + " is not fused here");
        }
        if (res) {
            if (rb == 0.0f) return ns_fail("ns_image_up_silo_conv: a residual weighted 0");
            if (ra / rb != 1.0f) po.append_eltwise(dnnl::algorithm::eltwise_linear, ra / rb, 0.0f);
            add_at = po.len();
            po.append_binary(dnnl::algorithm::binary_add, rmd);
            if (rb != 1.0f) po.append_eltwise(dnnl::algorithm::eltwise_linear, rb, 0.0f);
        }
        attr.set_post_ops(po);
        const memory::desc wany({Co, Ci, k, k}, F16, tag::any), bmd({Co}, F32, tag::a);
        auto pd = dnnl::convolution_forward::primitive_desc(d.eng, dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_direct, smd, wany,
                                                            bias ? bmd : memory::desc(), dmd, {stride, stride}, {pt, pl}, {pb, pr}, attr);
        if (d.prims.size() >= MAX_PRIMS) {
            d.st.wait();                           // none of them may still be running
            d.prims.clear();
        }
        it = d.prims.emplace(key, Prim{dnnl::convolution_forward(pd), pd.src_desc(), pd.weights_desc(), pd.dst_desc(), pd.scratchpad_desc(), prelu_at, add_at}).first;
    }
    Prim& p = it->second;
    if (p.src != smd || p.dst != dmd) return ns_fail("ns_image_up_silo_conv: oneDNN wants another layout for the activations");
    const size_t need = p.scratch.get_size();
    if (need > d.scratch_cap) {
        d.st.wait();
        if (d.scratch) sycl::free(d.scratch, g->q);
        d.scratch = sycl::malloc_device(need, g->q);
        d.scratch_cap = d.scratch ? need : 0;
        if (!d.scratch) return ns_fail("ns_image_up_silo_conv: no scratchpad of " + std::to_string(need >> 20) + " MiB");
    }
    auto& layouts = d.weights[w];
    memory* wm = nullptr;
    for (auto& m : layouts)
        if (m.get_desc() == p.wei) wm = &m;
    if (!wm) {
        memory src_w = mem(memory::desc({Co, Ci, k, k}, F16, tag::oihw), d.eng, w);
        memory nw(p.wei, d.eng);
        dnnl::reorder(src_w, nw).execute(d.st, src_w, nw);
        layouts.push_back(nw);
        wm = &layouts.back();
    }
    std::unordered_map<int, memory> args{{DNNL_ARG_SRC, mem(p.src, d.eng, x)}, {DNNL_ARG_WEIGHTS, *wm}, {DNNL_ARG_DST, mem(p.dst, d.eng, out)}};
    if (need) args.insert({DNNL_ARG_SCRATCHPAD, mem(p.scratch, d.eng, d.scratch)});
    if (bias) args.insert({DNNL_ARG_BIAS, mem(memory::desc({Co}, F32, tag::a), d.eng, bias)});
    if (p.prelu_at >= 0) args.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP(p.prelu_at) | DNNL_ARG_WEIGHTS, mem(pmd, d.eng, slopes)});
    if (p.add_at >= 0) args.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP(p.add_at) | DNNL_ARG_SRC_1, mem(rmd, d.eng, res)});
    p.p.execute(d.st, args);
    return 0;
    } catch (const dnnl::error& e) { return ns_fail(std::string("oneDNN (upscale conv): ") + e.what());
    NS_CATCH
}

}  // extern "C"
