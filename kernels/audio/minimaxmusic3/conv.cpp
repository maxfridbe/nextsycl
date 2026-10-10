// conv.cpp: the decoder's 1-D convolutions on oneDNN (the shared nsd_conv1d is a plain loop - seconds a window).
// Float32 signals [B, C, L] and weights as the checkpoint has them; oneDNN may compute in a narrower type
// (NS_MM3_FPMATH: f16, bf16, tf32; the default strict: float32, exact and as fast here). Primitives are made
// once a shape; each weight is reordered to oneDNN's preferred layout once (keyed by its pointer: the weights live as
// long as the engine).
#include "ns.h"
#include "ns_internal.hpp"
#include "mm3.h"

#include <sycl/sycl.hpp>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>

#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <tuple>
#include <unordered_map>

namespace {
using dnnl::memory;
using tag = memory::format_tag;

struct Prim {
    dnnl::primitive p;
    memory::desc src, wei, dst;
};

struct Dnn {
    dnnl::engine eng;
    dnnl::stream st;
    // (transposed, B, Ci, L, Co, K, stride, dil, left pad, right pad) -> the primitive
    std::map<std::tuple<int, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>, Prim> prims;
    // (weight pointer, primitive key's transposed / Ci / Co / K) -> the weight in the primitive's layout
    std::map<std::tuple<const void*, int, int64_t, int64_t, int64_t>, memory> weights;
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

dnnl::fpmath_mode fpmath() {
    const char* v = std::getenv("NS_MM3_FPMATH");
    if (!v || !std::strcmp(v, "strict")) return dnnl::fpmath_mode::strict;
    if (!std::strcmp(v, "f16")) return dnnl::fpmath_mode::f16;
    if (!std::strcmp(v, "bf16")) return dnnl::fpmath_mode::bf16;
    if (!std::strcmp(v, "tf32")) return dnnl::fpmath_mode::tf32;
    return dnnl::fpmath_mode::strict;
}

memory mem(const memory::desc& md, const dnnl::engine& eng, const void* p) {
    return dnnl::sycl_interop::make_memory(md, eng, dnnl::sycl_interop::memory_kind::usm, const_cast<void*>(p));
}

int run(ns_gpu* g, bool tr, const float* x, int64_t B, int64_t Ci, int64_t L, const float* w, int64_t Co, int64_t K, const float* bias,
        int64_t stride, int64_t dil, int64_t pl, int64_t pr, float* out, int64_t Lo) {
    Dnn& d = dnn(g);
    auto key = std::make_tuple((int) tr, B, Ci, L, Co, K, stride, dil, pl, pr);
    auto it = d.prims.find(key);
    if (it == d.prims.end()) {
        memory::desc src({B, Ci, L}, memory::data_type::f32, tag::ncw);
        memory::desc dst({B, Co, Lo}, memory::data_type::f32, tag::ncw);
        memory::desc wany({Co, Ci, K}, memory::data_type::f32, tag::any);
        memory::desc bmd({Co}, memory::data_type::f32, tag::a);
        dnnl::primitive_attr attr;
        attr.set_fpmath_mode(fpmath());
        Prim made;
        if (!tr) {
            auto pd = dnnl::convolution_forward::primitive_desc(d.eng, dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_direct, src, wany,
                                                                bias ? bmd : memory::desc(), dst, {stride}, {dil - 1}, {pl}, {pr}, attr);
            made = Prim{dnnl::convolution_forward(pd), pd.src_desc(), pd.weights_desc(), pd.dst_desc()};
        } else {
            auto pd = dnnl::deconvolution_forward::primitive_desc(d.eng, dnnl::prop_kind::forward_inference, dnnl::algorithm::deconvolution_direct, src,
                                                                  wany, bias ? bmd : memory::desc(), dst, {stride}, {pl}, {pr}, attr);
            made = Prim{dnnl::deconvolution_forward(pd), pd.src_desc(), pd.weights_desc(), pd.dst_desc()};
        }
        it = d.prims.emplace(key, made).first;
    }
    Prim& p = it->second;
    if (p.src != memory::desc({B, Ci, L}, memory::data_type::f32, tag::ncw) || p.dst != memory::desc({B, Co, Lo}, memory::data_type::f32, tag::ncw)) {
        return ns_fail("ns_audio_mm3_conv: oneDNN wants another layout for the signal");
    }
    auto wk = std::make_tuple((const void*) w, (int) tr, Ci, Co, K);
    auto wi = d.weights.find(wk);
    if (wi == d.weights.end() || wi->second.get_desc() != p.wei) {
        // the checkpoint's layout: [Co, Ci, K] (oiw), transposed [Ci, Co, K] (bac: logical o, i, w stored i, o, w)
        memory::desc plain({Co, Ci, K}, memory::data_type::f32, tr ? tag::bac : tag::oiw);
        memory pw = mem(plain, d.eng, w);
        memory nw(p.wei, d.eng);
        dnnl::reorder(pw, nw).execute(d.st, pw, nw);
        wi = d.weights.insert_or_assign(wk, nw).first;
    }
    std::unordered_map<int, memory> args{{DNNL_ARG_SRC, mem(p.src, d.eng, x)}, {DNNL_ARG_WEIGHTS, wi->second}, {DNNL_ARG_DST, mem(p.dst, d.eng, out)}};
    if (bias) args.insert({DNNL_ARG_BIAS, mem(memory::desc({Co}, memory::data_type::f32, tag::a), d.eng, bias)});
    p.p.execute(d.st, args);
    return 0;
}
}  // namespace

extern "C" {

int ns_audio_mm3_conv1d(ns_gpu* g, const float* x, int64_t B, int64_t Ci, int64_t L, const float* w, int64_t Co, int64_t K, const float* bias,
                        int64_t stride, int64_t dil, int64_t pad, float* out, int64_t Lo) {
    NS_TRY
    if (B <= 0 || Lo <= 0) return 0;
    return run(g, false, x, B, Ci, L, w, Co, K, bias, stride, dil, pad, pad, out, Lo);
    } catch (const dnnl::error& e) { return ns_fail(std::string("oneDNN: ") + e.what());
    NS_CATCH
}

int ns_audio_mm3_conv_transpose1d(ns_gpu* g, const float* x, int64_t B, int64_t Ci, int64_t L, const float* w, int64_t Co, int64_t K,
                                  const float* bias, int64_t stride, int64_t pad, float* out, int64_t Lo) {
    NS_TRY
    if (B <= 0 || Lo <= 0) return 0;
    return run(g, true, x, B, Ci, L, w, Co, K, bias, stride, 1, pad, pad, out, Lo);
    } catch (const dnnl::error& e) { return ns_fail(std::string("oneDNN: ") + e.what());
    NS_CATCH
}

int ns_audio_mm3_conv1d_lr(ns_gpu* g, int transposed, const float* x, int64_t B, int64_t Ci, int64_t L, const float* w, int64_t Co, int64_t K,
                           const float* bias, int64_t stride, int64_t dil, int64_t pad_l, int64_t pad_r, float* out, int64_t Lo) {
    NS_TRY
    if (B <= 0 || Lo <= 0) return 0;
    return run(g, transposed != 0, x, B, Ci, L, w, Co, K, bias, stride, transposed ? 1 : dil, pad_l, pad_r, out, Lo);
    } catch (const dnnl::error& e) { return ns_fail(std::string("oneDNN: ") + e.what());
    NS_CATCH
}

}  // extern "C"
