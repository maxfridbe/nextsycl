// example.cpp: the llm template engine's one kernel. The pattern every engine's kernels follow: SYCL only, on the
// GPU's own queue (ns_internal.hpp's ns_gpu), behind a C function that returns 0 or -1 with the reason in
// ns_last_error() (NS_TRY / NS_CATCH), declared in the engine's own header. Copy this directory to start a port.
#include "ns.h"
#include "ns_internal.hpp"
#include "example.h"

#include <sycl/sycl.hpp>

extern "C" int ns_llm_example_scale(ns_gpu* g, float* x, int64_t n, float s) {
    NS_TRY
    if (n < 0) return ns_fail("ns_llm_example_scale: a negative count");
    if (n == 0) return 0;
    g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) { x[i] *= s; });
    return 0;
    NS_CATCH
}
