// ns_internal.hpp: what the C ABI's translation units share (not part of the ABI).
#pragma once
#include <sycl/sycl.hpp>
#include <string>

struct ns_gpu {
    sycl::device dev;
    sycl::context ctx;   // this GPU alone (see ns.h)
    sycl::queue q;
};

// sets the thread's error (ns_last_error) and returns -1
int ns_fail(const std::string& what);

#define NS_TRY try {
#define NS_CATCH } catch (const sycl::exception& e) { return ns_fail(std::string("sycl: ") + e.what()); } \
                   catch (const std::exception& e) { return ns_fail(e.what()); }
