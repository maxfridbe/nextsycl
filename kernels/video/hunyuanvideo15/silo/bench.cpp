// bench.cpp: what the B70's matrix engine gives a plain product (silo.h, ns_video_hv15_silo_bench_gemm) - the yardstick
// for the DiT's attention and products: oneDNN's matmul alone on buffers of the size asked, half x half -> half or
// int8 x int8 -> int32, timed over `iters` runs after two to warm up. The values are whatever the buffers held (the
// time does not depend on them).
#include "ns.h"
#include "ns_internal.hpp"
#include "silo.h"

#include <sycl/sycl.hpp>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>

#include <chrono>
#include <string>

extern "C" int ns_video_hv15_silo_bench_gemm(ns_gpu* g, int kind, int64_t M, int64_t N, int64_t K, int iters, double* ms) {
    NS_TRY
    if (M <= 0 || N <= 0 || K <= 0 || iters <= 0 || iters > 1000 || M * K > (int64_t) 1 << 32 || N * K > (int64_t) 1 << 32 ||
        M * N > (int64_t) 1 << 31) {
        return ns_fail("bench_gemm: sizes out of range");
    }
    using dnnl::memory;
    using dt = memory::data_type;
    const bool i8 = kind == 1;
    const dt a_dt = i8 ? dt::s8 : dt::f16, c_dt = i8 ? dt::s32 : dt::f16;
    const size_t ab = i8 ? 1 : 2, cb = i8 ? 4 : 2;
    auto eng = dnnl::sycl_interop::make_engine(g->dev, g->ctx);
    auto st = dnnl::sycl_interop::make_stream(eng, g->q);
    void* a = sycl::malloc_device((size_t) (M * K) * ab, g->q);
    void* b = sycl::malloc_device((size_t) (N * K) * ab, g->q);
    void* c = sycl::malloc_device((size_t) (M * N) * cb, g->q);
    if (!a || !b || !c) {
        if (a) sycl::free(a, g->q);
        if (b) sycl::free(b, g->q);
        if (c) sycl::free(c, g->q);
        return ns_fail("bench_gemm: out of device memory");
    }
    g->q.memset(a, 0, (size_t) (M * K) * ab);
    g->q.memset(b, 0, (size_t) (N * K) * ab);
    g->q.wait();
    memory::desc ad({M, K}, a_dt, memory::format_tag::ab), bd({K, N}, a_dt, memory::dims{1, K}), cd({M, N}, c_dt, memory::format_tag::ab);
    dnnl::matmul mm(dnnl::matmul::primitive_desc(eng, ad, bd, cd));
    auto mk = [&](const memory::desc& d, void* p) { return dnnl::sycl_interop::make_memory(d, eng, dnnl::sycl_interop::memory_kind::usm, p); };
    auto am = mk(ad, a), bm = mk(bd, b), cm = mk(cd, c);
    auto run = [&] { mm.execute(st, {{DNNL_ARG_SRC, am}, {DNNL_ARG_WEIGHTS, bm}, {DNNL_ARG_DST, cm}}); };
    run();
    run();
    st.wait();
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) run();
    st.wait();
    *ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / iters;
    sycl::free(a, g->q);
    sycl::free(b, g->q);
    sycl::free(c, g->q);
    return 0;
    NS_CATCH
}
