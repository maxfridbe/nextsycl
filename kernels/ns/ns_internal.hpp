// ns_internal.hpp: what the C ABI's translation units share (not part of the ABI).
#pragma once
#include <sycl/sycl.hpp>
#include <array>
#include <vector>
#include <string>

struct ns_gpu {
    sycl::device dev;
    sycl::context ctx;   // this GPU alone (see ns.h)
    sycl::queue q;
    sycl::queue cq;      // copies that overlap q's work (ns_stream_copy); ordered against q on the device only
    sycl::queue cq2;     // a second copy lane: the other PCIe direction runs beside the first
    std::vector<sycl::event> ev = std::vector<sycl::event>(1 << 16);   // tickets (ns_mark, ns_stream_copy, ns_stamp), % size
    int64_t next_ev = 0;
};

// sets the thread's error (ns_last_error) and returns -1
int ns_fail(const std::string& what);

#define NS_TRY try {
#define NS_CATCH } catch (const sycl::exception& e) { return ns_fail(std::string("sycl: ") + e.what()); } \
                   catch (const std::exception& e) { return ns_fail(e.what()); }
