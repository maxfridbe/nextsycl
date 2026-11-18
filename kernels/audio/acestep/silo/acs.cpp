// acs.cpp: ACE-Step 1.5's silo - the Oobleck VAE's 1-D convolutions on oneDNN (acs.h). Started from MiniMax Music 3's
// decoder convolutions (kernels/audio/minimaxmusic3/conv.cpp), copied, not shared: a card's state is guarded for the
// whole call (the engine and the front door's threads), and emptied when the VAE's weights go.
#include "ns.h"
#include "ns_internal.hpp"
#include "acs.h"

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
    memory::desc src, wei, dst, scratch;
};

// the most shapes kept: a song's windows are of a few lengths (the full window, the last, the first), each ~70
// convolutions; past this many all are dropped and made again
constexpr size_t MAX_PRIMS = 512;

struct Dnn {
    std::mutex mu;  // one call at a time on the card: the maps, the scratchpad
    dnnl::engine eng;
    dnnl::stream st;
    // (transposed, biased, Ci, L, Co, K, stride, dil, pad) -> the primitive
    std::map<std::tuple<int, int, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>, Prim> prims;
    // (weight pointer, transposed, Ci, Co, K) -> the weight in the primitive's layout
    std::map<std::tuple<const void*, int, int64_t, int64_t, int64_t>, memory> weights;
    // the scratchpad every primitive uses (one stream), grown to the largest asked
    void* scratch = nullptr;
    size_t scratch_cap = 0;
    // (transposed weight pointer, its bias pointer, Ci, Co, K, stride) -> its phases as one convolution's weights and
    // bias (ns_acs_conv_transpose1d), on the card
    struct Split {
        float* w;
        float* b;
    };
    std::map<std::tuple<const void*, const void*, int64_t, int64_t, int64_t, int64_t>, Split> splits;
};

std::mutex mu;
std::map<ns_gpu*, std::unique_ptr<Dnn>> ctxs;

Dnn& dnn(ns_gpu* g) {
    std::lock_guard<std::mutex> l(mu);
    auto& d = ctxs[g];
    if (!d) {
        auto made = std::make_unique<Dnn>();
        made->eng = dnnl::sycl_interop::make_engine(g->dev, g->ctx);
        made->st = dnnl::sycl_interop::make_stream(made->eng, g->q);
        d = std::move(made);
    }
    return *d;
}

// NS_ACE_VAE_FPMATH: strict (default: float32, as the shared loop), tf32, bf16, f16 (oneDNN may compute narrower)
dnnl::fpmath_mode fpmath() {
    const char* v = std::getenv("NS_ACE_VAE_FPMATH");
    if (!v || !std::strcmp(v, "strict")) return dnnl::fpmath_mode::strict;
    if (!std::strcmp(v, "tf32")) return dnnl::fpmath_mode::tf32;
    if (!std::strcmp(v, "bf16")) return dnnl::fpmath_mode::bf16;
    if (!std::strcmp(v, "f16")) return dnnl::fpmath_mode::f16;
    return dnnl::fpmath_mode::strict;
}

memory mem(const memory::desc& md, const dnnl::engine& eng, const void* p) {
    return dnnl::sycl_interop::make_memory(md, eng, dnnl::sycl_interop::memory_kind::usm, const_cast<void*>(p));
}

// one primitive's run (d.mu held)
int run(Dnn& d, ns_gpu* g, bool tr, const float* x, int64_t Ci, int64_t L, const float* w, int64_t Co, int64_t K, const float* bias,
        int64_t stride, int64_t dil, int64_t pad, float* out, int64_t Lo) {
    const memory::desc src({1, Ci, L}, memory::data_type::f32, tag::ncw);
    const memory::desc dst({1, Co, Lo}, memory::data_type::f32, tag::ncw);
    auto key = std::make_tuple((int) tr, (int) (bias != nullptr), Ci, L, Co, K, stride, dil, pad);
    auto it = d.prims.find(key);
    if (it == d.prims.end()) {
        memory::desc wany({Co, Ci, K}, memory::data_type::f32, tag::any);
        memory::desc bmd({Co}, memory::data_type::f32, tag::a);
        dnnl::primitive_attr attr;
        attr.set_fpmath_mode(fpmath());
        attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
        Prim made;
        if (!tr) {
            auto pd = dnnl::convolution_forward::primitive_desc(d.eng, dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_direct, src, wany,
                                                                bias ? bmd : memory::desc(), dst, {stride}, {dil - 1}, {pad}, {pad}, attr);
            made = Prim{dnnl::convolution_forward(pd), pd.src_desc(), pd.weights_desc(), pd.dst_desc(), pd.scratchpad_desc()};
        } else {
            auto pd = dnnl::deconvolution_forward::primitive_desc(d.eng, dnnl::prop_kind::forward_inference, dnnl::algorithm::deconvolution_direct, src,
                                                                  wany, bias ? bmd : memory::desc(), dst, {stride}, {pad}, {pad}, attr);
            made = Prim{dnnl::deconvolution_forward(pd), pd.src_desc(), pd.weights_desc(), pd.dst_desc(), pd.scratchpad_desc()};
        }
        if (d.prims.size() >= MAX_PRIMS) d.prims.clear();
        it = d.prims.emplace(key, made).first;
    }
    Prim& p = it->second;
    if (p.src != src || p.dst != dst) return ns_fail("ns_acs_conv1d: oneDNN wants another layout for the signal");
    const size_t need = p.scratch.get_size();
    if (need > d.scratch_cap) {
        d.st.wait();  // nothing queued may still use the old one
        if (d.scratch) sycl::free(d.scratch, g->q);
        d.scratch = sycl::malloc_device(need, g->q);
        d.scratch_cap = d.scratch ? need : 0;
        if (!d.scratch) return ns_fail("ns_acs_conv1d: the device refused a scratchpad of " + std::to_string(need >> 20) + " MiB");
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
    if (need) args.insert({DNNL_ARG_SCRATCHPAD, mem(p.scratch, d.eng, d.scratch)});
    if (bias) args.insert({DNNL_ARG_BIAS, mem(memory::desc({Co}, memory::data_type::f32, tag::a), d.eng, bias)});
    p.p.execute(d.st, args);
    return 0;
}

// the card's memory given back (d.mu held): the scratchpad, the weights' copies; with `prims`, the primitives too.
// The card's state is emptied, not removed: another engine's call on the card may hold it (its weights are made again
// on its next call, from the checkpoint's copies)
void empty(Dnn& d, ns_gpu* g, bool prims) {
    d.st.wait();
    if (d.scratch) sycl::free(d.scratch, g->q);
    d.scratch = nullptr;
    d.scratch_cap = 0;
    d.weights.clear();
    for (auto& [k, sp] : d.splits) {
        sycl::free(sp.w, g->q);
        if (sp.b) sycl::free(sp.b, g->q);
    }
    d.splits.clear();
    if (prims) d.prims.clear();
}

Dnn* find(ns_gpu* g) {
    std::lock_guard<std::mutex> l(mu);
    auto it = ctxs.find(g);
    return it == ctxs.end() ? nullptr : it->second.get();
}
}  // namespace

extern "C" {

int ns_acs_supported(ns_gpu* g) {
    try {
        (void) dnn(g);
        return 1;
    } catch (...) {
        return 0;
    }
}

int ns_acs_conv1d(ns_gpu* g, int transposed, const float* x, int64_t Ci, int64_t L, const float* w, int64_t Co, int64_t K, const float* bias,
                  int64_t stride, int64_t dil, int64_t pad, float* out, int64_t Lo) {
    NS_TRY
    if (Ci <= 0 || L <= 0 || Co <= 0 || Lo <= 0) return 0;
    if (transposed && dil != 1) return ns_fail("ns_acs_conv1d: a transposed convolution with a dilation");
    Dnn& d = dnn(g);
    std::lock_guard<std::mutex> lock(d.mu);
    return run(d, g, transposed != 0, x, Ci, L, w, Co, K, bias, stride, dil, pad, out, Lo);
    } catch (const dnnl::error& e) { return ns_fail(std::string("oneDNN: ") + e.what());
    NS_CATCH
}

int ns_acs_conv_transpose1d(ns_gpu* g, const float* x, int64_t Ci, int64_t L, const float* w, int64_t Co, int64_t K, const float* bias,
                            int64_t stride, int64_t pad, float* tmp, int64_t tmp_floats, float* out, int64_t Lo) {
    NS_TRY
    if (Ci <= 0 || L <= 0 || Co <= 0 || Lo <= 0) return 0;
    const int64_t s = stride, J = (K + s - 1) / s, Lq = L + J - 1;
    Dnn& d = dnn(g);
    std::lock_guard<std::mutex> lock(d.mu);
    // the phases need m + pad within [0, s Lq): pad < s and Lo + pad <= s Lq (Oobleck's: pad = ceil(s / 2), K = 2 s)
    if (!tmp || tmp_floats < s * Co * Lq || pad < 0 || pad >= s || Lo + pad > s * Lq) {
        return run(d, g, true, x, Ci, L, w, Co, K, bias, stride, 1, pad, out, Lo);
    }
    auto sk = std::make_tuple((const void*) w, (const void*) bias, Ci, Co, K, s);
    auto si = d.splits.find(sk);
    if (si == d.splits.end()) {
        // ws[(r Co + o), i, t] = w[i, o, r + s (J - 1 - t)] (zero past K), bs[r Co + o] = bias[o]
        float* ws = sycl::malloc_device<float>((size_t) (s * Co * Ci * J), g->q);
        float* bs = bias ? sycl::malloc_device<float>((size_t) (s * Co), g->q) : nullptr;
        if (!ws || (bias && !bs)) {
            if (ws) sycl::free(ws, g->q);
            if (bs) sycl::free(bs, g->q);
            return ns_fail("ns_acs_conv_transpose1d: the device refused the phases' weights");
        }
        g->q.parallel_for(sycl::range<1>((size_t) (s * Co * Ci * J)), [=](sycl::id<1> id) {
            const int64_t n = id[0], t = n % J, i = (n / J) % Ci, ro = n / (J * Ci), o = ro % Co, r = ro / Co;
            const int64_t k = r + s * (J - 1 - t);
            ws[n] = k < K ? w[(i * Co + o) * K + k] : 0.0f;
        });
        if (bias) g->q.parallel_for(sycl::range<1>((size_t) (s * Co)), [=](sycl::id<1> id) { bs[id[0]] = bias[id[0] % Co]; });
        si = d.splits.emplace(sk, Dnn::Split{ws, bs}).first;
    }
    // tmp[(r Co + o), q] = bs + sum_i,t ws[(r Co + o), i, t] x[i, q - (J - 1) + t]: q over [0, L + J - 1)
    int rc = run(d, g, false, x, Ci, L, si->second.w, s * Co, J, si->second.b, 1, 1, J - 1, tmp, Lq);
    if (rc) return rc;
    // out[o, m] = tmp[((m + pad) % s) Co + o, (m + pad) / s]
    const float* tp = tmp;
    g->q.parallel_for(sycl::range<2>((size_t) Co, (size_t) Lo), [=](sycl::id<2> id) {
        const int64_t o = id[0], m = id[1], mp = m + pad;
        out[o * Lo + m] = tp[((mp % s) * Co + o) * Lq + mp / s];
    });
    return 0;
    } catch (const dnnl::error& e) { return ns_fail(std::string("oneDNN: ") + e.what());
    NS_CATCH
}

int ns_acs_held(ns_gpu* g, int64_t* weights, int64_t* scratch) {
    NS_TRY
    *weights = 0;
    *scratch = 0;
    Dnn* d = find(g);
    if (!d) return 0;
    std::lock_guard<std::mutex> lock(d->mu);
    for (auto& [k, m] : d->weights) *weights += (int64_t) m.get_desc().get_size();
    for (auto& [k, sp] : d->splits) {
        const int64_t Ci = std::get<2>(k), Co = std::get<3>(k), K = std::get<4>(k), s = std::get<5>(k);
        *weights += 4 * (s * Co * Ci * ((K + s - 1) / s) + (sp.b ? s * Co : 0));
    }
    *scratch = (int64_t) d->scratch_cap;
    return 0;
    } catch (const dnnl::error& e) { return ns_fail(std::string("oneDNN: ") + e.what());
    NS_CATCH
}

int ns_acs_trim(ns_gpu* g) {
    NS_TRY
    Dnn* d = find(g);
    if (!d) return 0;
    std::lock_guard<std::mutex> lock(d->mu);
    empty(*d, g, false);
    return 0;
    } catch (const dnnl::error& e) { return ns_fail(std::string("oneDNN: ") + e.what());
    NS_CATCH
}

int ns_acs_release(ns_gpu* g) {
    NS_TRY
    Dnn* d = find(g);
    if (!d) return 0;
    std::lock_guard<std::mutex> lock(d->mu);
    empty(*d, g, true);
    return 0;
    } catch (const dnnl::error& e) { return ns_fail(std::string("oneDNN: ") + e.what());
    NS_CATCH
}

}  // extern "C"
