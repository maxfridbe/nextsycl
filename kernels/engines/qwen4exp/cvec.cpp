// cvec.cpp: a control vector on the residual streams (the projection modes: llama.cpp's control vectors with the
// experimental-speed-projection package's --cvec-mode / --cvec-dir) - Strata's cvec kernel (sycl/src/kernels/cuda/
// cvec.dp.cpp at intel-arc-0.1.40, its arithmetic unchanged) with the tables on the stage's own GPU context (Strata's
// live on dpct's per-device contexts). After layer l's FFN write every hc stream h of every token becomes
//   project:  h <- h - s_l (h . v_l) v_l      add:  h <- h + d_l
// for the steered layers. A device flag switches it per request, so the graphs are the same either way.
#include "qwen_internal.hpp"

#include <dpct/dpct.hpp>

#include <string>

using namespace qw;

struct Cv {
    float* dir = nullptr;   // n_layer x n_embd: project = the unit v_l, add = d_l
    float* s = nullptr;     // n_layer: project = s_l, add = 1; 0 = not steered
    int* on = nullptr;      // 0: the stock model
    int mode = 0;           // 0 project, 1 add
    std::vector<bool> steered;
};

namespace {

constexpr int THREADS = 256;
constexpr int MAXK = 16;

inline float sigmoidf_(float x) { return 1.0f / (1.0f + sycl::native::exp(-x)); }

// one work-group per (stream, token): the pending write, then h . v over the stream, then the update
void cvec_kernel(float* __restrict__ R, const float* __restrict__ dir, const float* __restrict__ s_l,
                 const int* __restrict__ on, int mode, int64_t layer, int n, int hc, int64_t r_ld,
                 const float* __restrict__ bo, int64_t bo_ld, const float* __restrict__ inj, int64_t inj_ld, int write,
                 sycl::nd_item<3> item, float* part) {
    const int c = item.get_group(2);
    const int64_t t = item.get_group(1);
    float* r = R + t * r_ld + (int64_t) c * n;
    const float s = s_l[layer];
    const bool steer = *on != 0 && s != 0.0f;
    if (!steer && !write) return;
    const float* v = dir + layer * n;
    const float w = write ? 2.0f * sigmoidf_(inj[t * inj_ld + c] / (float) hc) : 0.0f;
    const float* b = write ? bo + t * bo_ld : nullptr;
    float x[MAXK];
    float dot = 0.0f;
#pragma unroll
    for (int k = 0; k < MAXK; ++k) {
        const int d = item.get_local_id(2) + k * THREADS;
        if (d < n) {
            float xv = r[d];
            if (write) xv = sycl::fma(b[d], w, xv);
            x[k] = xv;
            if (steer && mode == 0) dot = sycl::fma(xv, v[d], dot);
        }
    }
    if (steer && mode == 0) {
        auto sg = item.get_sub_group();
        for (int o = 16; o > 0; o >>= 1) dot += sycl::permute_group_by_xor(sg, dot, o);
        if ((item.get_local_id(2) & 31) == 0) part[item.get_local_id(2) >> 5] = dot;
        item.barrier(sycl::access::fence_space::local_space);
        if (item.get_local_id(2) < 32) {
            float p = item.get_local_id(2) < THREADS / 32 ? part[item.get_local_id(2)] : 0.0f;
            for (int o = 16; o > 0; o >>= 1) p += sycl::permute_group_by_xor(sg, p, o);
            if (item.get_local_id(2) == 0) part[0] = p;
        }
        item.barrier(sycl::access::fence_space::local_space);
        dot = part[0] * s;
    }
#pragma unroll
    for (int k = 0; k < MAXK; ++k) {
        const int d = item.get_local_id(2) + k * THREADS;
        if (d < n) {
            float xv = x[k];
            if (steer) xv = mode == 0 ? sycl::fma(-dot, v[d], xv) : xv + v[d];
            r[d] = xv;
        }
    }
}

}  // namespace

namespace qw {

bool cvec_covers(const ns_qw* w, int64_t l) {
    return w->cv != nullptr && l >= 0 && l < (int64_t) w->cv->steered.size() && w->cv->steered[(size_t) l];
}

void cvec_apply(ns_qw* w, float* R, int64_t layer, int64_t T, int64_t r_ld, const float* bo, int64_t bo_ld,
                const float* inj, int64_t inj_ld, bool write) {
    const Cv* cv = w->cv;
    if (cv == nullptr || T < 1) return;
    const float* dir = cv->dir;
    const float* s = cv->s;
    const int* on = cv->on;
    const int mode = cv->mode;
    const int wr = write ? 1 : 0;
    w->q->submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> part(sycl::range<1>(THREADS / 32), h);
        h.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, (size_t) T, (size_t) HC * THREADS), sycl::range<3>(1, 1, THREADS)),
                       [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(32)]] {
                           cvec_kernel(R, dir, s, on, mode, layer, (int) N, (int) HC, r_ld, bo, bo_ld, inj, inj_ld, wr, item,
                                       part.get_multi_ptr<sycl::access::decorated::no>().get());
                       });
    });
}

void cvec_free(ns_qw* w) {
    Cv* cv = w->cv;
    if (cv == nullptr) return;
    for (void* p : {(void*) cv->dir, (void*) cv->s, (void*) cv->on})
        if (p) sycl::free(p, w->g->ctx);
    delete cv;
    w->cv = nullptr;
}

}  // namespace qw

extern "C" {

int ns_qw_cvec_set(ns_qw* w, const float* dir, const float* s, int n_layer, int mode) {
    NS_TRY
    if (n_layer != w->n_layer) return ns_fail("qwen cvec: a vector of another layer count");
    if (w->last_st != nullptr || w->pf != nullptr) return ns_fail("qwen cvec: set before any window");
    qw::cvec_free(w);
    auto* cv = new Cv();
    cv->mode = mode;
    auto& ctx = w->g->ctx;
    cv->dir = (float*) sycl::malloc_device((size_t) n_layer * N * 4, w->g->dev, ctx);
    cv->s = (float*) sycl::malloc_device((size_t) n_layer * 4, w->g->dev, ctx);
    cv->on = (int*) sycl::malloc_device(4, w->g->dev, ctx);
    w->cv = cv;
    if (!cv->dir || !cv->s || !cv->on) { qw::cvec_free(w); return ns_fail("qwen cvec: device memory"); }
    const int one = 1;
    w->q->memcpy(cv->dir, dir, (size_t) n_layer * N * 4);
    w->q->memcpy(cv->s, s, (size_t) n_layer * 4);
    w->q->memcpy(cv->on, &one, 4);
    w->q->wait_and_throw();
    cv->steered.assign((size_t) n_layer, false);
    for (int l = 0; l < n_layer; ++l) cv->steered[(size_t) l] = s[l] != 0.0f;
    return 0;
    NS_CATCH
}

int ns_qw_cvec_enable(ns_qw* w, int on) {
    NS_TRY
    if (w->cv == nullptr) return 0;
    w->q->memcpy(w->cv->on, &on, 4).wait();
    return 0;
    NS_CATCH
}

}  // extern "C"
