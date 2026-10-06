// ns.cpp: the devices, memory and copies of the C ABI (ns.h). Kernels are called through their own wrappers
// (ns_*.cpp beside this), each taking the ns_gpu's queue.
#include "ns.h"
#include "ns_internal.hpp"

#include <sycl/sycl.hpp>

#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

thread_local std::string g_err;

int fail(const std::string& what) { return ns_fail(what); }

std::vector<sycl::device> gpus() {
    std::vector<sycl::device> v;
    for (const auto& p : sycl::platform::get_platforms()) {
        if (p.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
        for (const auto& d : p.get_devices(sycl::info::device_type::gpu)) v.push_back(d);
    }
    return v;
}

}  // namespace

int ns_fail(const std::string& what) {
    g_err = what;
    return -1;
}

extern "C" {

const char* ns_last_error(void) { return g_err.c_str(); }
const char* ns_version(void) { return "nextsycl 0.1"; }

int ns_gpu_count(void) {
    NS_TRY
    return (int) gpus().size();
    NS_CATCH
}

int ns_gpu_name(int index, char* buf, size_t len) {
    NS_TRY
    auto v = gpus();
    if (index < 0 || index >= (int) v.size()) return fail("no GPU " + std::to_string(index));
    std::snprintf(buf, len, "%s", v[index].get_info<sycl::info::device::name>().c_str());
    return 0;
    NS_CATCH
}

int ns_gpu_open(int index, ns_gpu** out) {
    NS_TRY
    auto v = gpus();
    if (index < 0 || index >= (int) v.size()) return fail("no GPU " + std::to_string(index) + " (" + std::to_string(v.size()) + " found)");
    sycl::context ctx(v[index]);
    *out = new ns_gpu{v[index], ctx, sycl::queue(ctx, v[index], sycl::property::queue::in_order())};
    return 0;
    NS_CATCH
}

void ns_gpu_close(ns_gpu* g) {
    if (!g) return;
    try { g->q.wait(); } catch (...) {}
    delete g;
}

void* ns_gpu_queue(ns_gpu* g) { return &g->q; }

int ns_gpu_memory(ns_gpu* g, uint64_t* total, int64_t* free_bytes) {
    NS_TRY
    *total = g->dev.get_info<sycl::info::device::global_mem_size>();
    *free_bytes = -1;
    if (g->dev.has(sycl::aspect::ext_intel_free_memory)) {
        try { *free_bytes = (int64_t) g->dev.get_info<sycl::ext::intel::info::device::free_memory>(); } catch (...) {}
    }
    return 0;
    NS_CATCH
}

int ns_alloc(ns_gpu* g, size_t bytes, void** out) {
    NS_TRY
    *out = sycl::malloc_device(bytes, g->q);
    if (!*out) return fail("device allocation of " + std::to_string(bytes) + " bytes failed");
    return 0;
    NS_CATCH
}

int ns_free(ns_gpu* g, void* p) {
    NS_TRY
    sycl::free(p, g->q);
    return 0;
    NS_CATCH
}

int ns_alloc_host(ns_gpu* g, size_t bytes, void** out) {
    NS_TRY
    *out = sycl::malloc_host(bytes, g->q);
    if (!*out) return fail("pinned host allocation of " + std::to_string(bytes) + " bytes failed");
    return 0;
    NS_CATCH
}

int ns_free_host(ns_gpu* g, void* p) { return ns_free(g, p); }

int ns_copy_to(ns_gpu* g, void* dst, const void* src, size_t bytes) {
    NS_TRY
    g->q.memcpy(dst, src, bytes);
    return 0;
    NS_CATCH
}

int ns_copy_from(ns_gpu* g, void* dst, const void* src, size_t bytes) {
    NS_TRY
    g->q.memcpy(dst, src, bytes);
    return 0;
    NS_CATCH
}

int ns_copy_dev(ns_gpu* g, void* dst, const void* src, size_t bytes) {
    NS_TRY
    g->q.memcpy(dst, src, bytes);
    return 0;
    NS_CATCH
}

int ns_fill(ns_gpu* g, void* dst, uint8_t value, size_t bytes) {
    NS_TRY
    g->q.memset(dst, value, bytes);
    return 0;
    NS_CATCH
}

int ns_sync(ns_gpu* g) {
    NS_TRY
    g->q.wait_and_throw();
    return 0;
    NS_CATCH
}

int ns_copy_peer(ns_gpu* to, void* dst, ns_gpu* from, const void* src, size_t bytes, void* staging) {
    NS_TRY
    from->q.memcpy(staging, src, bytes).wait();
    to->q.memcpy(dst, staging, bytes).wait();
    return 0;
    NS_CATCH
}

}  // extern "C"
