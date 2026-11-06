// attn.cpp: the DiT's window attention in bfloat16 - oneDNN's fused attention graph (MatMul, scale, SoftMax,
// MatMul) over B windows of S rows read in place from the packed qkv rows. The shared attention runs in half (its
// fast path; bfloat16 it copies to half): a value past half's 65,504 in v becomes inf and the whole output NaN - now
// and then on late blocks of real clips. The reference runs it in bfloat16 (flash attention); so does this. A
// compiled partition a shape.
#include "ns.h"
#include "ns_internal.hpp"
#include "silo.h"

#include <sycl/sycl.hpp>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_graph.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>

#include <cmath>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

namespace {
using lt = dnnl::graph::logical_tensor;

struct Part {
    dnnl::graph::compiled_partition cp;
    std::vector<lt> in;
    std::vector<int> slot;   // 0 q, 1 k, 2 scale, 3 v
    lt out;
};

struct Ctx {
    dnnl::engine eng;
    dnnl::stream st;
    std::map<std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t>, Part> parts;
    float scale = 1.0f;
};

std::mutex mu;
std::map<ns_gpu*, std::unique_ptr<Ctx>> ctxs;

Ctx& ctx(ns_gpu* g) {
    std::lock_guard<std::mutex> l(mu);
    auto& c = ctxs[g];
    if (!c) {
        c = std::make_unique<Ctx>();
        c->eng = dnnl::sycl_interop::make_engine(g->dev, g->ctx);
        c->st = dnnl::sycl_interop::make_stream(c->eng, g->q);
    }
    return *c;
}

Part& part(Ctx& c, int64_t B, int64_t S, int64_t H, int64_t D, int64_t stride) {
    auto key = std::make_tuple(B, S, H, D, stride);
    auto it = c.parts.find(key);
    if (it != c.parts.end()) return it->second;
    using namespace dnnl::graph;
    const auto bf = lt::data_type::bf16, f32 = lt::data_type::f32;
    const lt::dims d4 {B, H, S, D}, sc4 {B, H, S, S};
    const lt::dims in_st {S * stride, D, stride, 1}, out_st {S * H * D, D, H * D, 1};
    size_t id = 0;
    lt q(id++, bf, d4, in_st), k(id++, bf, d4, in_st);
    lt scale(id++, f32, lt::dims {}, lt::layout_type::strided, lt::property_type::host_scalar);
    lt v(id++, bf, d4, in_st);
    lt score(id++, f32, sc4, lt::layout_type::strided), scaled(id++, f32, sc4, lt::layout_type::strided);
    lt probs(id++, bf, sc4, lt::layout_type::strided), out(id++, bf, d4, out_st);
    op bmm1(id++, op::kind::MatMul, {q, k}, {score}, "scores");
    bmm1.set_attr<bool>(op::attr::transpose_b, true);
    op mul(id++, op::kind::Multiply, {score, scale}, {scaled}, "scale");
    op sm(id++, op::kind::SoftMax, {scaled}, {probs}, "softmax");
    sm.set_attr<int64_t>(op::attr::axis, -1);
    op bmm2(id++, op::kind::MatMul, {probs, v}, {out}, "values");
    graph gr(dnnl::engine::kind::gpu);
    gr.add_op(bmm1); gr.add_op(mul); gr.add_op(sm); gr.add_op(bmm2);
    gr.finalize();
    auto ps = gr.get_partitions();
    if (ps.size() != 1 || !ps[0].is_supported()) throw std::runtime_error("oneDNN does not take the bfloat16 attention as one partition");
    const lt mine[4] = {q, k, scale, v};
    Part p;
    for (const auto& port : ps[0].get_input_ports())
        for (int i = 0; i < 4; ++i)
            if (mine[i].get_id() == port.get_id()) { p.in.push_back(mine[i]); p.slot.push_back(i); }
    if (p.in.size() != 4) throw std::runtime_error("oneDNN's bfloat16 attention partition has unexpected inputs");
    p.cp = ps[0].compile(p.in, {out}, c.eng);
    p.out = p.cp.query_logical_tensor(out.get_id());
    if (c.parts.size() >= 64) {
        c.st.wait();                                 // none of them may still be running
        c.parts.clear();
    }
    return c.parts.emplace(key, std::move(p)).first->second;
}
}  // namespace

extern "C" {

int ns_image_seedvr2_silo_attention(ns_gpu* g, const void* q, const void* k, const void* v, int64_t B, int64_t S, int64_t H, int64_t D,
                                    int64_t stride, void* out) {
    NS_TRY
    if (B <= 0 || S <= 0) return 0;
    Ctx& c = ctx(g);
    Part& p = part(c, B, S, H, D, stride);
    c.scale = 1.0f / std::sqrt((float) D);
    const void* const handles[4] = {q, k, nullptr, v};
    std::vector<dnnl::graph::tensor> in;
    for (size_t i = 0; i < p.in.size(); ++i) {
        if (p.slot[i] == 2) in.push_back(dnnl::graph::tensor::make_scalar_tensor(p.in[i], &c.scale));
        else in.emplace_back(p.in[i], c.eng, const_cast<void*>(handles[p.slot[i]]));
    }
    p.cp.execute(c.st, in, {dnnl::graph::tensor(p.out, c.eng, out)});
    return 0;
    } catch (const dnnl::error& e) { return ns_fail(std::string("oneDNN: ") + e.what());
    NS_CATCH
}

}  // extern "C"
