// vcs.cpp: VoxCPM2's silo (vcs.h) - the sampler's guided Euler step on the device, and the command graph a patch's
// evaluations are recorded in.
#include "ns.h"
#include "ns_internal.hpp"
#include "vcs.h"

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/math.hpp>

namespace {
namespace X = sycl::ext::oneapi::experimental;
using Graph = X::command_graph<X::graph_state::modifiable>;
using Exec = X::command_graph<X::graph_state::executable>;
constexpr int64_t NMAX = 4096;
}  // namespace

extern "C" {

int ns_vcs_supported(int64_t n) {
    return n > 0 && n <= NMAX ? 1 : 0;
}

// the audio kernels build with the compiler's fast floating point: this step rounds as the host's float32 does - no
// fused multiply-adds, no reassociated sums (precise), and a correctly rounded division (Intel's fdiv_rn: the
// device's own divide is not; the other operations are the hardware's, which round to nearest)
#pragma float_control(precise, on, push)
int ns_vcs_cfg_euler(ns_gpu* g, float* x, const float* v, const int32_t* order, int64_t n, float dt, float cfg) {
    NS_TRY
    if (!ns_vcs_supported(n)) return ns_fail("ns_vcs_cfg_euler: 1 to 4,096 values");
    // one work-item: the two sums in the host's order, so the step is the host's to the bit
    g->q.single_task([=] {
#pragma clang fp contract(off) reassociate(off)
        const float* pos = v;
        const float* neg = v + n;
        float dot = -0.0f, nn = -0.0f;
        for (int64_t j = 0; j < n; ++j) {
            const int64_t i = order[j];
            dot += pos[i] * neg[i];
        }
        for (int64_t j = 0; j < n; ++j) {
            const int64_t i = order[j];
            nn += neg[i] * neg[i];
        }
        nn += 1e-8f;
        const float st = sycl::ext::intel::math::fdiv_rn(dot, nn);
        for (int64_t i = 0; i < n; ++i) {
            const float ns = neg[i] * st;
            const float vi = ns + cfg * (pos[i] - ns);
            x[i] = x[i] - dt * vi;
        }
    });
    return 0;
    NS_CATCH
}
#pragma float_control(pop)

int ns_vcs_graph_begin(ns_gpu* g, void** graph) {
    NS_TRY
    auto* gr = new Graph(g->q.get_context(), g->q.get_device());
    gr->begin_recording(g->q);
    *graph = gr;
    return 0;
    NS_CATCH
}

int ns_vcs_graph_end(ns_gpu* g, void* graph, void** exec) {
    NS_TRY
    auto* gr = (Graph*) graph;
    gr->end_recording(g->q);
    *exec = new Exec(gr->finalize());
    delete gr;
    return 0;
    NS_CATCH
}

int ns_vcs_graph_launch(ns_gpu* g, void* exec) {
    NS_TRY
    g->q.ext_oneapi_graph(*(Exec*) exec);
    return 0;
    NS_CATCH
}

void ns_vcs_graph_free(void* exec) {
    delete (Exec*) exec;
}

}  // extern "C"
