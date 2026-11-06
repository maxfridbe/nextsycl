// dnn.cpp: FlashVSR's products on oneDNN - the DiT's linear layers and the LQ projection's and decoder's convolutions,
// half in, float32 accumulation, with what follows each fused into it (bias, ReLU / GELU, the DiT's gate and residual
// sum, the MemBlocks' residual), so no product makes a second pass over its output. As kernels/image/seedvr2/conv.cpp:
// a primitive a shape, each weight reordered once a layout (keyed by its pointer: `ns_image_flashvsr_dnn_clear` forgets
// them when an engine's weights go), one user scratchpad a card for every primitive, the shapes kept bounded. A weight's
// layout is the one oneDNN chose for its first product (its shape's canonical layout), asked of every later primitive of
// that weight shape: each other layout would be another copy of the weights (2.4 GB for the DiT's) - a run at 2560 x
// 1408 ran out of the card's memory so.
#include "ns.h"
#include "ns_internal.hpp"
#include "fv.h"

#include <sycl/sycl.hpp>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>

#include <cstdlib>
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

struct Prim {
    dnnl::primitive p;
    memory::desc src, wei, dst, scratch;
};

constexpr size_t MAX_PRIMS = 512;

struct Dnn {
    dnnl::engine eng;
    dnnl::stream st;
    std::map<std::vector<int64_t>, Prim> prims;
    // (weight pointer) -> its reorders, one a layout asked for
    std::map<const void*, std::vector<memory>> weights;
    // (kind, weight dims) -> the layout its first primitive chose
    std::map<std::vector<int64_t>, memory::desc> canon;
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

// the scratchpad a primitive wants, grown (the queue drained first: nothing queued may still use the old one)
bool scratch(ns_gpu* g, Dnn& d, size_t need) {
    if (need <= d.scratch_cap) return true;
    d.st.wait();
    if (d.scratch) sycl::free(d.scratch, g->q);
    d.scratch = sycl::malloc_device(need, g->q);
    d.scratch_cap = d.scratch ? need : 0;
    return d.scratch != nullptr;
}

// w in its plain layout `plain` reordered to `want` once (kept by its pointer); the plain buffer itself when it is the
// layout wanted (no copy)
memory& weight(Dnn& d, const void* w, const memory::desc& plain, const memory::desc& want) {
    auto& layouts = d.weights[w];
    if (want == plain) {
        if (layouts.empty()) layouts.push_back(mem(plain, d.eng, w));
        return layouts.front();
    }
    for (auto& m : layouts)
        if (m.get_desc() == want) return m;
    memory pw = mem(plain, d.eng, w);
    memory nw(want, d.eng);
    dnnl::reorder(pw, nw).execute(d.st, pw, nw);
    layouts.push_back(nw);
    return layouts.back();
}

void bound(Dnn& d) {
    if (d.prims.size() < MAX_PRIMS) return;
    d.st.wait();                                   // none of them may still be running
    d.prims.clear();
}
}  // namespace

extern "C" {

// Forget the reordered weights and the primitives on this card (an engine's weights are gone: a later engine's buffer
// at the same address must not find their reorders)
int ns_image_flashvsr_dnn_clear(ns_gpu* g) {
    NS_TRY
    Dnn& d = dnn(g);
    d.st.wait();
    d.weights.clear();
    d.prims.clear();
    d.canon.clear();
    return 0;
    } catch (const dnnl::error& e) { return ns_fail(std::string("oneDNN: ") + e.what());
    NS_CATCH
}

int ns_image_flashvsr_gemm(ns_gpu* g, const void* x, int64_t M, int64_t K, int64_t lda, const void* w, int64_t N, const float* bias, int act,
                           const float* gate, int accum, void* out, int64_t ldo, int out_f32) {
    NS_TRY
    if (M <= 0) return 0;
    if (lda < K || ldo < N) return ns_fail("ns_image_flashvsr_gemm: a row stride shorter than its row");
    if (accum && !out_f32) return ns_fail("ns_image_flashvsr_gemm: a residual sum is float32");
    Dnn& d = dnn(g);
    const auto odt = out_f32 ? F32 : F16;
    const memory::desc smd({M, K}, F16, {lda, 1}), dmd({M, N}, odt, {ldo, 1});
    const memory::desc wplain({K, N}, F16, tag::ba), bmd({1, N}, F32, tag::ab), gmd({1, N}, F32, tag::ab);
    std::vector<int64_t> key{0, M, K, N, lda, ldo, out_f32, bias ? 1 : 0, act, gate ? 1 : 0, accum};
    auto it = d.prims.find(key);
    if (it == d.prims.end()) {
        dnnl::primitive_attr attr;
        attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
        dnnl::post_ops po;
        if (act == 1) po.append_eltwise(dnnl::algorithm::eltwise_relu, 0.0f, 0.0f);
        if (act == 2) po.append_eltwise(dnnl::algorithm::eltwise_gelu_tanh, 0.0f, 0.0f);
        if (gate) po.append_binary(dnnl::algorithm::binary_mul, gmd);
        if (accum) po.append_sum(1.0f);
        attr.set_post_ops(po);
        // the products' weights as they are ([N, K], read transposed: no reordered copy beside them - the DiT's would be
        // 2.4 GB more); NS_FVSR_WREORDER=1: oneDNN's own layout, reordered once a weight
        static const bool reorder = std::getenv("NS_FVSR_WREORDER") && std::string(std::getenv("NS_FVSR_WREORDER")) == "1";
        const memory::desc wany = reorder ? memory::desc({K, N}, F16, tag::any) : wplain;
        const std::vector<int64_t> ck{0, K, N};
        auto make = [&](const memory::desc& wd) { return dnnl::matmul::primitive_desc(d.eng, smd, wd, bias ? bmd : memory::desc(), dmd, attr); };
        dnnl::matmul::primitive_desc pd;
        auto c = d.canon.find(ck);
        try {
            pd = c != d.canon.end() ? make(c->second) : make(wany);
        } catch (const dnnl::error&) {
            pd = make(wany);                       // this shape cannot take the canonical layout: its own
        }
        if (c == d.canon.end()) d.canon[ck] = pd.weights_desc();
        bound(d);
        it = d.prims.emplace(key, Prim{dnnl::matmul(pd), pd.src_desc(), pd.weights_desc(), pd.dst_desc(), pd.scratchpad_desc()}).first;
    }
    Prim& p = it->second;
    if (!scratch(g, d, p.scratch.get_size())) return ns_fail("ns_image_flashvsr_gemm: the device refused a scratchpad");
    memory& wm = weight(d, w, wplain, p.wei);
    std::unordered_map<int, memory> args{{DNNL_ARG_SRC, mem(p.src, d.eng, x)}, {DNNL_ARG_WEIGHTS, wm}, {DNNL_ARG_DST, mem(p.dst, d.eng, out)}};
    if (p.scratch.get_size()) args.insert({DNNL_ARG_SCRATCHPAD, mem(p.scratch, d.eng, d.scratch)});
    if (bias) args.insert({DNNL_ARG_BIAS, mem(bmd, d.eng, bias)});
    if (gate) {
        const int at = act ? 1 : 0;
        args.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP(at) | DNNL_ARG_SRC_1, mem(gmd, d.eng, gate)});
    }
    p.p.execute(d.st, args);
    return 0;
    } catch (const dnnl::error& e) {
        return ns_fail(std::string("oneDNN (gemm ") + std::to_string(M) + " x " + std::to_string(K) + " x " + std::to_string(N) + "): " + e.what());
    NS_CATCH
}

int ns_image_flashvsr_conv(ns_gpu* g, const void* x, int64_t N, int64_t H, int64_t W, int64_t Ci, int64_t ldx, const void* w, int64_t Co, int64_t k,
                           int64_t pad, const float* bias, int accum, int act, void* out) {
    NS_TRY
    if (N <= 0) return 0;
    if (ldx < Ci) return ns_fail("ns_image_flashvsr_conv: a pixel stride shorter than its channels");
    const int64_t Ho = H + 2 * pad - k + 1, Wo = W + 2 * pad - k + 1;
    if (Ho <= 0 || Wo <= 0) return ns_fail("ns_image_flashvsr_conv: the picture is smaller than the kernel");
    Dnn& d = dnn(g);
    const memory::desc smd({N, Ci, H, W}, F16, {H * W * ldx, 1, W * ldx, ldx}), dmd({N, Co, Ho, Wo}, F16, tag::nhwc);
    const memory::desc wplain({Co, Ci, k, k}, F16, tag::oihw), bmd({Co}, F32, tag::a);
    std::vector<int64_t> key{1, N, H, W, Ci, ldx, Co, k, pad, bias ? 1 : 0, accum, act};
    auto it = d.prims.find(key);
    if (it == d.prims.end()) {
        dnnl::primitive_attr attr;
        attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
        dnnl::post_ops po;
        if (accum) po.append_sum(1.0f);
        if (act == 1) po.append_eltwise(dnnl::algorithm::eltwise_relu, 0.0f, 0.0f);
        attr.set_post_ops(po);
        const memory::desc wany({Co, Ci, k, k}, F16, tag::any);
        const std::vector<int64_t> ck{1, Co, Ci, k};
        auto make = [&](const memory::desc& wd) {
            return dnnl::convolution_forward::primitive_desc(d.eng, dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_direct, smd, wd,
                                                             bias ? bmd : memory::desc(), dmd, {1, 1}, {pad, pad}, {pad, pad}, attr);
        };
        dnnl::convolution_forward::primitive_desc pd;
        auto c = d.canon.find(ck);
        try {
            pd = c != d.canon.end() ? make(c->second) : make(wany);
        } catch (const dnnl::error&) {
            pd = make(wany);
        }
        if (c == d.canon.end()) d.canon[ck] = pd.weights_desc();
        bound(d);
        it = d.prims.emplace(key, Prim{dnnl::convolution_forward(pd), pd.src_desc(), pd.weights_desc(), pd.dst_desc(), pd.scratchpad_desc()}).first;
    }
    Prim& p = it->second;
    if (p.src != smd || p.dst != dmd) return ns_fail("ns_image_flashvsr_conv: oneDNN wants another layout for the frames");
    if (!scratch(g, d, p.scratch.get_size())) return ns_fail("ns_image_flashvsr_conv: the device refused a scratchpad");
    memory& wm = weight(d, w, wplain, p.wei);
    std::unordered_map<int, memory> args{{DNNL_ARG_SRC, mem(p.src, d.eng, x)}, {DNNL_ARG_WEIGHTS, wm}, {DNNL_ARG_DST, mem(p.dst, d.eng, out)}};
    if (p.scratch.get_size()) args.insert({DNNL_ARG_SCRATCHPAD, mem(p.scratch, d.eng, d.scratch)});
    if (bias) args.insert({DNNL_ARG_BIAS, mem(bmd, d.eng, bias)});
    p.p.execute(d.st, args);
    return 0;
    } catch (const dnnl::error& e) {
        return ns_fail(std::string("oneDNN (conv ") + std::to_string(N) + " x " + std::to_string(H) + " x " + std::to_string(W) + " x " + std::to_string(Ci) + " -> " +
                       std::to_string(Co) + "): " + e.what());
    NS_CATCH
}

}  // extern "C"
