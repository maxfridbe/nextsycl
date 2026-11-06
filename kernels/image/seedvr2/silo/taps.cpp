// taps.cpp: the VAE's causal convolutions as kt 2-D convolutions accumulated (oneDNN, a sum post-op after the first
// tap). The engine's own path (conv.cpp) copies the kept frames and the slice into one volume before each 3-D
// convolution and copies the last frames out after it - at 1728x960 most of a convolution's memory traffic again; here
// each tap reads its frames where they are. A primitive a shape and tap kind (first: bias, later: sum), each tap's
// weights reordered once (keyed by the weight's pointer and the tap; every layout kept), one user scratchpad.
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

struct Prim {
    dnnl::convolution_forward p;
    memory::desc src, wei, dst, scratch;
};

constexpr size_t MAX_PRIMS = 256;

struct Dnn {
    dnnl::engine eng;
    dnnl::stream st;
    std::map<std::vector<int64_t>, Prim> prims;
    // (weight pointer, tap) -> its reordered copies, one a layout
    std::map<std::pair<const void*, int64_t>, std::vector<memory>> weights;
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

// one tap over n frames that sit one after another: out (n frames) = [bias +] conv2d(src) [+ out]
int tap(ns_gpu* g, Dnn& d, const void* src, int64_t n, int64_t H, int64_t W, int64_t Ci, const void* w, int64_t Co, int64_t kt, int64_t j,
        int64_t kh, int64_t kw, int64_t ph, int64_t pw, const float* bias, bool sum, void* out) {
    const memory::desc smd({n, Ci, H, W}, BF, tag::nhwc), dmd({n, Co, H, W}, BF, tag::nhwc);
    std::vector<int64_t> key{n, H, W, Ci, Co, kh, kw, ph, pw, bias ? 1 : 0, sum ? 1 : 0};
    auto it = d.prims.find(key);
    if (it == d.prims.end()) {
        dnnl::primitive_attr attr;
        attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
        if (sum) {
            dnnl::post_ops po;
            po.append_sum(1.0f);
            attr.set_post_ops(po);
        }
        const memory::desc wany({Co, Ci, kh, kw}, BF, tag::any), bmd({Co}, memory::data_type::f32, tag::a);
        auto pd = dnnl::convolution_forward::primitive_desc(d.eng, dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_direct, smd,
                                                            wany, bias ? bmd : memory::desc(), dmd, {1, 1}, {ph, pw}, {ph, pw}, attr);
        if (d.prims.size() >= MAX_PRIMS) {
            d.st.wait();                           // none of them may still be running
            d.prims.clear();
        }
        it = d.prims.emplace(key, Prim{dnnl::convolution_forward(pd), pd.src_desc(), pd.weights_desc(), pd.dst_desc(), pd.scratchpad_desc()}).first;
    }
    Prim& p = it->second;
    if (p.src != smd || p.dst != dmd) return ns_fail("ns_image_seedvr2_silo_conv_taps: oneDNN wants another layout for the frames");
    const size_t need = p.scratch.get_size();
    if (need > d.scratch_cap) {
        d.st.wait();
        if (d.scratch) sycl::free(d.scratch, g->q);
        d.scratch = sycl::malloc_device(need, g->q);
        d.scratch_cap = d.scratch ? need : 0;
        if (!d.scratch) return ns_fail("ns_image_seedvr2_silo_conv_taps: no scratchpad of " + std::to_string(need >> 20) + " MiB");
    }
    auto& layouts = d.weights[{w, j}];
    memory* wm = nullptr;
    for (auto& m : layouts)
        if (m.get_desc() == p.wei) wm = &m;
    if (!wm) {
        // tap j of [Co, Ci, kt, kh, kw]: a strided view of the 3-D weight, reordered into the primitive's layout
        const memory::desc view({Co, Ci, kh, kw}, BF, memory::dims{Ci * kt * kh * kw, kt * kh * kw, kw, 1});
        memory src_w = mem(view, d.eng, static_cast<const uint16_t*>(w) + j * kh * kw);
        memory nw(p.wei, d.eng);
        dnnl::reorder(src_w, nw).execute(d.st, src_w, nw);
        layouts.push_back(nw);
        wm = &layouts.back();
    }
    std::unordered_map<int, memory> args{{DNNL_ARG_SRC, mem(p.src, d.eng, src)}, {DNNL_ARG_WEIGHTS, *wm}, {DNNL_ARG_DST, mem(p.dst, d.eng, out)}};
    if (need) args.insert({DNNL_ARG_SCRATCHPAD, mem(p.scratch, d.eng, d.scratch)});
    if (bias) args.insert({DNNL_ARG_BIAS, mem(memory::desc({Co}, memory::data_type::f32, tag::a), d.eng, bias)});
    p.p.execute(d.st, args);
    return 0;
}
}  // namespace

extern "C" {

int ns_image_seedvr2_silo_supported(ns_gpu* g) {
    return g ? 1 : 0;
}

int ns_image_seedvr2_silo_conv_taps(ns_gpu* g, const void* const* frames, int64_t nframes, int64_t H, int64_t W, int64_t Ci, const void* w,
                                    int64_t Co, int64_t kt, int64_t kh, int64_t kw, int64_t ph, int64_t pw, const float* bias, void* out) {
    NS_TRY
    const int64_t To = nframes - kt + 1;
    if (To <= 0 || H <= 0 || W <= 0) return ns_fail("ns_image_seedvr2_silo_conv_taps: fewer frames than taps");
    Dnn& d = dnn(g);
    const size_t fin = (size_t) (H * W * Ci) * 2, fout = (size_t) (H * W * Co) * 2;
    for (int64_t j = 0; j < kt; ++j) {
        // out frames t read frames t + j: runs where those frames are contiguous go together
        int64_t t = 0;
        while (t < To) {
            int64_t n = 1;
            while (t + n < To && static_cast<const char*>(frames[t + j + n]) == static_cast<const char*>(frames[t + j + n - 1]) + fin) ++n;
            void* o = static_cast<char*>(out) + (size_t) t * fout;
            const int rc = tap(g, d, frames[t + j], n, H, W, Ci, w, Co, kt, j, kh, kw, ph, pw, j == 0 ? bias : nullptr, j > 0, o);
            if (rc != 0) return rc;
            t += n;
        }
    }
    return 0;
    } catch (const dnnl::error& e) { return ns_fail(std::string("oneDNN: ") + e.what());
    NS_CATCH
}

}  // extern "C"
