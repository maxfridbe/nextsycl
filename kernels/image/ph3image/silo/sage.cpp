// sage.cpp: the P-H3-Image silo's SageAttention (silo.h). The kernel is H3's (libnextsycl-flash.so's nsflash_sage_square,
// ARK's int8 prefill on sycl-tla), the same the shared nsd_attention calls for long sequences; what is this silo's own are
// the passes around it, which in the shared form take 14 ms of a 34 ms attention at P-H3-Image's 8 MP (8,318 tokens,
// 56 heads of 128):
//   k's mean    one work-item a column walking all 8,318 rows (7,168 work-items: the card mostly idle) -> each work-item
//               sums a 128-row slice of 8 columns (16-byte loads), the slices added with atomics
//   q, k, v     three launches of an element a work-item -> one launch, a work-group a (tensor, head, 64-row block),
//               16-byte loads and stores, the block's values kept in registers between the max and the quantization
//   the result  half [H, S, D] -> the activations [S, H * D], an element a work-item -> 16 bytes a work-item
// The arithmetic is the shared form's (the scale m / 127, round half away from zero, clamp to 127); only the order of the
// mean's sum differs.
#include "ns.h"
#include "ns_internal.hpp"
#include "silo.h"

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <map>
#include <memory>
#include <mutex>
#include <string>

namespace {
constexpr int64_t BLOCK = 64;      // rows a scale (the kernel's scale block)
constexpr int64_t SLICE = 128;     // rows a work-item sums for k's mean
constexpr int WG = 256;

using sage_fn = int (*)(void*, const int8_t*, const int8_t*, const void*, void*, const float*, const float*, int, int64_t, int64_t, int64_t, float);
using err_fn = const char* (*)();

struct Flash {
    sage_fn sage = nullptr;
    err_fn err = nullptr;
    bool tried = false;
};
Flash flash;
std::mutex flash_mu;

// libnextsycl-flash.so as the shared kernels find it: NS_FLASH_LIB (the program's embedded copy), else beside the kind's
// library (this silo is in dist/silo/, the kind's library and flash one level up)
bool load_flash() {
    std::lock_guard<std::mutex> lk(flash_mu);
    if (flash.tried) return flash.sage != nullptr;
    flash.tried = true;
    std::string path;
    if (const char* e = std::getenv("NS_FLASH_LIB"); e && *e) {
        path = e;
    } else {
        Dl_info me;
        if (!dladdr((void*) &load_flash, &me) || !me.dli_fname) return false;
        std::string self = me.dli_fname;
        const size_t slash = self.rfind('/');
        path = (slash == std::string::npos ? std::string(".") : self.substr(0, slash)) + "/../libnextsycl-flash.so";
    }
    void* h = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h) return false;
    flash.sage = (sage_fn) dlsym(h, "nsflash_sage_square");
    flash.err = (err_fn) dlsym(h, "nsflash_error");
    if (!flash.sage || !flash.err) flash.sage = nullptr;
    return flash.sage != nullptr;
}

// per GPU: the int8 q and k, the scales, the half v and result, grown to the largest sequence seen
struct Buffers {
    int8_t* iq = nullptr;
    int8_t* ik = nullptr;
    float* sc = nullptr;
    sycl::half* hv = nullptr;
    sycl::half* ao = nullptr;
    size_t n = 0, nsc = 0;
};
std::mutex mu;
std::map<ns_gpu*, std::unique_ptr<Buffers>> bufs;

Buffers& buffers(ns_gpu* g, size_t n, size_t nsc) {
    auto& b = bufs[g];
    if (!b) b = std::make_unique<Buffers>();
    if (b->n < n || b->nsc < nsc) {
        g->q.wait();
        sycl::free(b->iq, g->q); sycl::free(b->ik, g->q); sycl::free(b->sc, g->q); sycl::free(b->hv, g->q); sycl::free(b->ao, g->q);
        b->iq = sycl::malloc_device<int8_t>(n, g->q);
        b->ik = sycl::malloc_device<int8_t>(n, g->q);
        b->sc = sycl::malloc_device<float>(nsc, g->q);
        b->hv = sycl::malloc_device<sycl::half>(n, g->q);
        b->ao = sycl::malloc_device<sycl::half>(n, g->q);
        if (!b->iq || !b->ik || !b->sc || !b->hv || !b->ao) {
            b->n = b->nsc = 0;
            throw std::runtime_error("ph3 silo: the attention's buffers do not fit the card");
        }
        b->n = n;
        b->nsc = nsc;
    }
    return *b;
}

inline float bf(uint16_t h) {
    return sycl::bit_cast<float>((uint32_t) h << 16);
}

inline uint16_t to_bf(float f) {
    uint32_t u = sycl::bit_cast<uint32_t>(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t) ((u >> 16) | 0x40);   // NaN stays NaN
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}
}  // namespace

extern "C" {

int ns_image_ph3_silo_sage_supported(ns_gpu* g, int64_t S, int64_t H, int64_t D) {
    (void) g;
    if (D != 128 || S < BLOCK || H <= 0 || S > INT32_MAX / D / H) return 0;
    return load_flash() ? 1 : 0;
}

int ns_image_ph3_silo_sage(ns_gpu* g, const void* qkv, int64_t S, int64_t H, int64_t D, int64_t stride, void* out) try {
    if (D != 128 || stride % 8 != 0 || !load_flash()) return ns_fail("ph3 silo sage: D 128, a row stride of whole 16-byte vectors, flash");
    std::lock_guard<std::mutex> lk(mu);
    sycl::queue& q = g->q;
    const int64_t HD = H * D, nb = (S + BLOCK - 1) / BLOCK;
    const size_t n = (size_t) S * HD;
    Buffers& b = buffers(g, n, (size_t) (2 * H * nb + HD));
    int8_t* iq = b.iq;
    int8_t* ik = b.ik;
    float* qs = b.sc;
    float* ks = b.sc + H * nb;
    float* ksum = b.sc + 2 * H * nb;
    sycl::half* hv = b.hv;
    sycl::half* ao = b.ao;
    const uint16_t* x = static_cast<const uint16_t*>(qkv);

    // k's column sums: a work-item a (slice of SLICE rows, 8 columns)
    q.memset(ksum, 0, (size_t) HD * sizeof(float));
    const int64_t vecs = HD / 8, slices = (S + SLICE - 1) / SLICE;
    q.parallel_for(sycl::range<2>((size_t) slices, (size_t) vecs), [=](sycl::id<2> id) {
        const int64_t s0 = (int64_t) id[0] * SLICE, s1 = sycl::min<int64_t>(S, s0 + SLICE);
        const int64_t c = (int64_t) id[1] * 8;
        float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int64_t t = s0; t < s1; ++t) {
            const sycl::vec<uint16_t, 8> v = *reinterpret_cast<const sycl::vec<uint16_t, 8>*>(x + t * stride + HD + c);
            for (int j = 0; j < 8; ++j) acc[j] += bf(v[j]);
        }
        for (int j = 0; j < 8; ++j) sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device>(ksum[c + j]).fetch_add(acc[j]);
    });
    // q and k - mean to int8 (a scale a head a 64-row block), v to half [H, S, D]: a work-group a (tensor, head, block),
    // its 64 x 128 values as 1,024 vectors of 8, four a work-item
    const float inv_s = 1.0f / (float) S;
    q.parallel_for(sycl::nd_range<1>((size_t) (3 * H * nb * WG), WG), [=](sycl::nd_item<1> it) {
        const int64_t gid = it.get_group(0);
        const int tensor = (int) (gid / (H * nb));
        const int64_t h = (gid / nb) % H, blk = gid % nb;
        const int64_t r0 = blk * BLOCK, rows = sycl::min<int64_t>(BLOCK, S - r0);
        const int lid = (int) it.get_local_id(0);
        constexpr int PER = (int) (BLOCK * 128 / 8 / WG);   // 4 vectors a work-item
        float vals[PER][8];
        float m = 0.0f;
        for (int k = 0; k < PER; ++k) {
            const int vi = lid + k * WG;
            const int64_t r = r0 + vi / 16, d = (vi % 16) * 8;
            if (vi / 16 >= rows) {
                for (int j = 0; j < 8; ++j) vals[k][j] = 0.0f;
                continue;
            }
            const sycl::vec<uint16_t, 8> v = *reinterpret_cast<const sycl::vec<uint16_t, 8>*>(x + r * stride + tensor * HD + h * D + d);
            for (int j = 0; j < 8; ++j) {
                float f = bf(v[j]);
                if (tensor == 1) f -= ksum[h * D + d + j] * inv_s;
                vals[k][j] = f;
                m = sycl::fmax(m, sycl::fabs(f));
            }
        }
        if (tensor == 2) {
            for (int k = 0; k < PER; ++k) {
                const int vi = lid + k * WG;
                if (vi / 16 >= rows) continue;
                const int64_t r = r0 + vi / 16, d = (vi % 16) * 8;
                sycl::vec<sycl::half, 8> o;
                for (int j = 0; j < 8; ++j) o[j] = (sycl::half) vals[k][j];
                *reinterpret_cast<sycl::vec<sycl::half, 8>*>(hv + (h * S + r) * D + d) = o;
            }
            return;
        }
        m = sycl::reduce_over_group(it.get_group(), m, sycl::maximum<float>());
        const float inv = m > 0.0f ? 127.0f / m : 0.0f;
        int8_t* dst = tensor == 0 ? iq : ik;
        if (lid == 0) (tensor == 0 ? qs : ks)[h * nb + blk] = m / 127.0f;
        for (int k = 0; k < PER; ++k) {
            const int vi = lid + k * WG;
            if (vi / 16 >= rows) continue;
            const int64_t r = r0 + vi / 16, d = (vi % 16) * 8;
            sycl::vec<int8_t, 8> o;
            for (int j = 0; j < 8; ++j) o[j] = (int8_t) sycl::clamp(sycl::round(vals[k][j] * inv), -127.0f, 127.0f);
            *reinterpret_cast<sycl::vec<int8_t, 8>*>(dst + (h * S + r) * D + d) = o;
        }
    });
    if (flash.sage(&q, iq, ik, hv, ao, qs, ks, (int) BLOCK, S, H, D, 1.0f / std::sqrt((float) D)) != 0) return ns_fail(flash.err());
    // the result [H, S, D] half -> [S, H * D] bfloat16, 8 values a work-item
    uint16_t* o = static_cast<uint16_t*>(out);
    q.parallel_for(sycl::range<2>((size_t) S, (size_t) (HD / 8)), [=](sycl::id<2> id) {
        const int64_t s = id[0], c = (int64_t) id[1] * 8, h = c / D, d = c % D;
        const sycl::vec<sycl::half, 8> v = *reinterpret_cast<const sycl::vec<sycl::half, 8>*>(ao + (h * S + s) * D + d);
        sycl::vec<uint16_t, 8> w;
        for (int j = 0; j < 8; ++j) w[j] = to_bf((float) v[j]);
        *reinterpret_cast<sycl::vec<uint16_t, 8>*>(o + s * HD + c) = w;
    });
    return 0;
} catch (const std::exception& e) {
    return ns_fail(std::string("ph3 silo sage: ") + e.what());
}

}  // extern "C"
