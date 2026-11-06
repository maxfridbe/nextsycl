// conv.cpp: the VAE's causal 3-D convolutions on oneDNN, bfloat16 channels-last volumes. The shared nsd_conv3d pads
// every side by k / 2 (zeros in front in time, where a causal convolution repeats its first frame or carries the
// last slice's frames); here time is never padded - the caller puts those frames in front - and the spatial padding
// is each side's own (the downsamplers pad right and bottom only). As kernels/audio/minimaxmusic3/conv.cpp: a
// primitive a shape, each weight reordered once a layout (keyed by its pointer: the weights live as long as the engine;
// every layout kept - replacing one would free it under a queued convolution still reading it), one
// user scratchpad for every primitive (each GPU primitive's own would be sized by the volume), the shapes kept bounded.
#include "ns.h"
#include "ns_internal.hpp"
#include "svr.h"

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

struct Prim {
    dnnl::convolution_forward p;
    memory::desc src, wei, dst, scratch;
};

// a clip's last slice and the single-picture path have shapes of their own: a few dozen primitives each
constexpr size_t MAX_PRIMS = 256;

struct Dnn {
    dnnl::engine eng;
    dnnl::stream st;
    std::map<std::vector<int64_t>, Prim> prims;
    // (weight pointer, its primitive's weight layout) -> the reordered weight
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

int ns_image_seedvr2_conv3d(ns_gpu* g, const void* x, int64_t T, int64_t H, int64_t W, int64_t Ci, const void* w, int64_t Co, int64_t kt,
                            int64_t kh, int64_t kw, int64_t st, int64_t sh, int64_t sw, int64_t ph0, int64_t ph1, int64_t pw0, int64_t pw1,
                            const float* bias, void* out) {
    NS_TRY
    if (T < kt || H <= 0 || W <= 0) return ns_fail("ns_image_seedvr2_conv3d: fewer frames than the kernel's depth");
    const int64_t To = (T - kt) / st + 1, Ho = (H + ph0 + ph1 - kh) / sh + 1, Wo = (W + pw0 + pw1 - kw) / sw + 1;
    if (Ho <= 0 || Wo <= 0) return ns_fail("ns_image_seedvr2_conv3d: the picture is smaller than the kernel");
    Dnn& d = dnn(g);
    const memory::desc smd({1, Ci, T, H, W}, BF, tag::ndhwc), dmd({1, Co, To, Ho, Wo}, BF, tag::ndhwc);
    std::vector<int64_t> key{T, H, W, Ci, Co, kt, kh, kw, st, sh, sw, ph0, ph1, pw0, pw1, bias ? 1 : 0};
    auto it = d.prims.find(key);
    if (it == d.prims.end()) {
        dnnl::primitive_attr attr;
        attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
        const memory::desc wany({Co, Ci, kt, kh, kw}, BF, tag::any), bmd({Co}, memory::data_type::f32, tag::a);
        auto pd = dnnl::convolution_forward::primitive_desc(d.eng, dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_direct, smd,
                                                            wany, bias ? bmd : memory::desc(), dmd, {st, sh, sw}, {0, ph0, pw0}, {0, ph1, pw1}, attr);
        if (d.prims.size() >= MAX_PRIMS) {
            d.st.wait();                           // none of them may still be running
            d.prims.clear();
        }
        it = d.prims.emplace(key, Prim{dnnl::convolution_forward(pd), pd.src_desc(), pd.weights_desc(), pd.dst_desc(), pd.scratchpad_desc()}).first;
    }
    Prim& p = it->second;
    if (p.src != smd || p.dst != dmd) return ns_fail("ns_image_seedvr2_conv3d: oneDNN wants another layout for the volume");
    const size_t need = p.scratch.get_size();
    if (need > d.scratch_cap) {
        d.st.wait();                               // nothing queued may still use the old one
        if (d.scratch) sycl::free(d.scratch, g->q);
        d.scratch = sycl::malloc_device(need, g->q);
        d.scratch_cap = d.scratch ? need : 0;
        if (!d.scratch) return ns_fail("ns_image_seedvr2_conv3d: the device refused a scratchpad of " + std::to_string(need >> 20) + " MiB");
    }
    auto& layouts = d.weights[w];
    memory* wm = nullptr;
    for (auto& m : layouts)
        if (m.get_desc() == p.wei) wm = &m;
    if (!wm) {
        memory pw = mem(memory::desc({Co, Ci, kt, kh, kw}, BF, tag::oidhw), d.eng, w);
        memory nw(p.wei, d.eng);
        dnnl::reorder(pw, nw).execute(d.st, pw, nw);
        layouts.push_back(nw);
        wm = &layouts.back();
    }
    std::unordered_map<int, memory> args{{DNNL_ARG_SRC, mem(p.src, d.eng, x)}, {DNNL_ARG_WEIGHTS, *wm}, {DNNL_ARG_DST, mem(p.dst, d.eng, out)}};
    if (need) args.insert({DNNL_ARG_SCRATCHPAD, mem(p.scratch, d.eng, d.scratch)});
    if (bias) args.insert({DNNL_ARG_BIAS, mem(memory::desc({Co}, memory::data_type::f32, tag::a), d.eng, bias)});
    p.p.execute(d.st, args);
    return 0;
    } catch (const dnnl::error& e) { return ns_fail(std::string("oneDNN: ") + e.what());
    NS_CATCH
}

}  // extern "C"
