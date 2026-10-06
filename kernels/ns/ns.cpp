// ns.cpp: the devices, memory and copies of the C ABI (ns.h). Kernels are called through their own wrappers
// (ns_*.cpp beside this), each taking the ns_gpu's queue.
#include "ns.h"
#include "ns_internal.hpp"

#include <sycl/sycl.hpp>
#include <cstdlib>

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

// the PCI address of GPU `index` ("0000:03:00.0") into buf; an empty string when the driver does not say
int ns_gpu_pci(int index, char* buf, size_t len) {
    NS_TRY
    auto v = gpus();
    if (index < 0 || index >= (int) v.size()) return fail("no GPU " + std::to_string(index));
    std::string a;
    if (v[index].has(sycl::aspect::ext_intel_pci_address)) a = v[index].get_info<sycl::ext::intel::info::device::pci_address>();
    std::snprintf(buf, len, "%s", a.c_str());
    return 0;
    NS_CATCH
}

int ns_gpu_open(int index, ns_gpu** out) {
    NS_TRY
    auto v = gpus();
    if (index < 0 || index >= (int) v.size()) return fail("no GPU " + std::to_string(index) + " (" + std::to_string(v.size()) + " found)");
    sycl::context ctx(v[index]);
    // NS_PROFILE=gpu: the queue keeps device timestamps (ns_stamp / ns_elapsed)
    const char* pf = getenv("NS_PROFILE");
    const bool stamps = pf && std::string(pf) == "gpu";
    sycl::property_list qp = stamps ? sycl::property_list{sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()}
                                    : sycl::property_list{sycl::property::queue::in_order()};
    *out = new ns_gpu{v[index], ctx, sycl::queue(ctx, v[index], qp),
                      sycl::queue(ctx, v[index], sycl::property::queue::in_order()),
                      sycl::queue(ctx, v[index], sycl::property::queue::in_order())};
    return 0;
    NS_CATCH
}

void ns_gpu_close(ns_gpu* g) {
    if (!g) return;
    try { g->q.wait(); g->cq.wait(); g->cq2.wait(); } catch (...) {}
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

// Copies beside the GPU's work. The order between the two queues is kept on the device (barriers with events):
// the host never waits on an event (Strata: host waits on queue events deadlocked under Level Zero v2).
static int64_t keep(ns_gpu* g, sycl::event e) {
    const int64_t t = g->next_ev++;
    g->ev[t % g->ev.size()] = e;
    return t;
}

// a ticket for everything submitted to the GPU's queue so far
int ns_mark(ns_gpu* g, int64_t* ticket) {
    NS_TRY
    *ticket = keep(g, g->q.ext_oneapi_submit_barrier());
    return 0;
    NS_CATCH
}

// bytes from src to dst on copy lane `lane` (0 or 1), after the `ndeps` tickets `deps`; its own ticket
int ns_stream_copy(ns_gpu* g, void* dst, const void* src, size_t bytes, const int64_t* after, int ndeps, int lane, int64_t* ticket) {
    NS_TRY
    std::vector<sycl::event> deps;
    for (int i = 0; i < ndeps; ++i) deps.push_back(g->ev[after[i] % g->ev.size()]);
    sycl::queue& q = lane ? g->cq2 : g->cq;
    *ticket = keep(g, q.memcpy(dst, src, bytes, deps));
    return 0;
    NS_CATCH
}

// a timestamp: a one-item kernel on the GPU's queue (its end is when the work before it ended); its ticket
int ns_stamp(ns_gpu* g, int64_t* ticket) {
    NS_TRY
    *ticket = keep(g, g->q.single_task([=]() {}));
    return 0;
    NS_CATCH
}

// nanoseconds between the ends of tickets t0 and t1 (stamps; waits for them - profiling only)
int ns_elapsed(ns_gpu* g, int64_t t0, int64_t t1, double* ns) {
    NS_TRY
    const auto& a = g->ev[t0 % g->ev.size()];
    const auto& b = g->ev[t1 % g->ev.size()];
    *ns = (double) b.get_profiling_info<sycl::info::event_profiling::command_end>() -
          (double) a.get_profiling_info<sycl::info::event_profiling::command_end>();
    return 0;
    NS_CATCH
}

// the GPU's queue waits (on the device) for ticket t
int ns_await(ns_gpu* g, int64_t t) {
    NS_TRY
    g->q.ext_oneapi_submit_barrier({g->ev[t % g->ev.size()]});
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
