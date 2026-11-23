// attn.cpp: HunyuanVideo-1.5's attention (silo.h, ns_video_hv15_silo_attention): the DiT's joint attention over one
// qkv buffer [S, 3 H D] (half; q, k after their norm and RoPE), the result half [S, H D] - the kernels are this silo's
// sycl-tla part (tla/attn.cpp -> libnextsycl-hunyuanvideo15-tla.so beside this library, opened on first use):
//   mode 0  half: ARK's half prefill kernel on the tiles asked (tile 0 = ARK's own, the shared kernels' flash)
//   mode 1  SageAttention v1: q and k to int8 here (a scale per head per 64 rows; k less its mean over the sequence,
//           which softmax does not see - the same shift for every key of a row), v and the result in place. The pass is
//           the MiniMax-H3-Image silo's (kernels/image/h3image/silo/sage.cpp) on half input: k's mean summed in
//           128-row slices of 8 columns, then a work-group a (tensor, head, 64-row block) with its values in registers
//           between the maximum and the quantization (scale = max / 127, round half away from zero, clamp to 127).
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
constexpr int64_t BLOCK = 64;   // rows a scale
constexpr int64_t SLICE = 128;  // rows a work-item sums for k's mean
constexpr int WG = 256;

using half_fn = int (*)(void*, const void*, const void*, const void*, void*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, float, int);
using sage_fn = int (*)(void*, const int8_t*, const int8_t*, const void*, void*, const float*, const float*, int, int64_t, int64_t, int64_t,
                        int64_t, int64_t, float);
using tiles_fn = int (*)();
using err_fn = const char* (*)();

struct Tla {
    half_fn half = nullptr;
    sage_fn sage = nullptr;
    tiles_fn tiles = nullptr;
    err_fn err = nullptr;
    bool tried = false;
};
Tla tla;
std::mutex tla_mu;

bool load_tla() {
    std::lock_guard<std::mutex> lk(tla_mu);
    if (tla.tried) return tla.half != nullptr;
    tla.tried = true;
    Dl_info me;
    if (!dladdr((void*) &load_tla, &me) || !me.dli_fname) return false;
    std::string self = me.dli_fname;
    const size_t slash = self.rfind('/');
    const std::string path = (slash == std::string::npos ? std::string(".") : self.substr(0, slash)) + "/libnextsycl-hunyuanvideo15-tla.so";
    void* h = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h) return false;
    tla.half = (half_fn) dlsym(h, "ns_video_hv15_tla_half");
    tla.sage = (sage_fn) dlsym(h, "ns_video_hv15_tla_sage");
    tla.tiles = (tiles_fn) dlsym(h, "ns_video_hv15_tla_tiles");
    tla.err = (err_fn) dlsym(h, "ns_video_hv15_tla_error");
    if (!tla.half || !tla.sage || !tla.tiles || !tla.err) tla.half = nullptr;
    return tla.half != nullptr;
}

// per GPU: int8 q and k [H, S, D] and their scales (+ k's column sums), grown to the longest sequence seen
struct Buffers {
    int8_t* iq = nullptr;
    int8_t* ik = nullptr;
    float* sc = nullptr;
    size_t n = 0, nsc = 0;
};
std::mutex mu;
std::map<ns_gpu*, std::unique_ptr<Buffers>> bufs;

Buffers& buffers(ns_gpu* g, size_t n, size_t nsc) {
    auto& b = bufs[g];
    if (!b) b = std::make_unique<Buffers>();
    if (b->n < n || b->nsc < nsc) {
        g->q.wait();
        sycl::free(b->iq, g->q); sycl::free(b->ik, g->q); sycl::free(b->sc, g->q);
        b->iq = sycl::malloc_device<int8_t>(n, g->q);
        b->ik = sycl::malloc_device<int8_t>(n, g->q);
        b->sc = sycl::malloc_device<float>(nsc, g->q);
        if (!b->iq || !b->ik || !b->sc) {
            b->n = b->nsc = 0;
            throw std::runtime_error("hunyuanvideo15 silo: the attention's buffers do not fit the card");
        }
        b->n = n;
        b->nsc = nsc;
    }
    return *b;
}
}  // namespace

extern "C" {

int ns_video_hv15_silo_attention_tiles(ns_gpu* g) {
    (void) g;
    return load_tla() ? tla.tiles() : 0;
}

int ns_video_hv15_silo_attention(ns_gpu* g, const void* qkv, int64_t S, int64_t H, int64_t D, int64_t stride, void* out, int64_t o_rs,
                                 int mode, int tile) {
    NS_TRY
    if (!load_tla()) return ns_fail("hunyuanvideo15 silo attention: no libnextsycl-hunyuanvideo15-tla.so beside the silo");
    if (D != 128 || S <= 0 || H <= 0 || stride % 8 != 0 || stride < 3 * H * D || o_rs < H * D) return ns_fail("hunyuanvideo15 silo attention: D 128, rows of whole 16-byte vectors");
    sycl::queue& q = g->q;
    const int64_t HD = H * D;
    const auto* x = static_cast<const sycl::half*>(qkv);
    const float scale = 1.0f / std::sqrt((float) D);
    if (mode == 0) {
        if (tla.half(&q, x, x + HD, x + 2 * HD, out, S, H, D, stride, stride, o_rs, scale, tile) != 0) return ns_fail(tla.err());
        return 0;
    }
    if (mode != 1) return ns_fail("hunyuanvideo15 silo attention: mode 0 (half) or 1 (sage)");
    std::lock_guard<std::mutex> lk(mu);
    const int64_t nb = (S + BLOCK - 1) / BLOCK;
    Buffers& b = buffers(g, (size_t) (S * HD), (size_t) (2 * H * nb + HD));
    int8_t* iq = b.iq;
    int8_t* ik = b.ik;
    float* qs = b.sc;
    float* ks = b.sc + H * nb;
    float* ksum = b.sc + 2 * H * nb;
    // k's column sums: a work-item a (slice of SLICE rows, 8 columns)
    q.memset(ksum, 0, (size_t) HD * sizeof(float));
    const int64_t vecs = HD / 8, slices = (S + SLICE - 1) / SLICE;
    q.parallel_for(sycl::range<2>((size_t) slices, (size_t) vecs), [=](sycl::id<2> id) {
        const int64_t s0 = (int64_t) id[0] * SLICE, s1 = sycl::min<int64_t>(S, s0 + SLICE);
        const int64_t c = (int64_t) id[1] * 8;
        float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int64_t t = s0; t < s1; ++t) {
            const sycl::vec<sycl::half, 8> v = *reinterpret_cast<const sycl::vec<sycl::half, 8>*>(x + t * stride + HD + c);
            for (int j = 0; j < 8; ++j) acc[j] += (float) v[j];
        }
        for (int j = 0; j < 8; ++j) sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device>(ksum[c + j]).fetch_add(acc[j]);
    });
    // q and k (less the mean) to int8, a work-group a (tensor, head, block): 64 x 128 values as 1,024 vectors of 8,
    // four a work-item
    const float inv_s = 1.0f / (float) S;
    q.parallel_for(sycl::nd_range<1>((size_t) (2 * H * nb * WG), WG), [=](sycl::nd_item<1> it) {
        const int64_t gid = it.get_group(0);
        const int tensor = (int) (gid / (H * nb));
        const int64_t h = (gid / nb) % H, blk = gid % nb;
        const int64_t r0 = blk * BLOCK, rows = sycl::min<int64_t>(BLOCK, S - r0);
        const int lid = (int) it.get_local_id(0);
        constexpr int PER = (int) (BLOCK * 128 / 8 / WG);
        float vals[PER][8];
        float m = 0.0f;
        for (int k = 0; k < PER; ++k) {
            const int vi = lid + k * WG;
            const int64_t r = r0 + vi / 16, d = (vi % 16) * 8;
            if (vi / 16 >= rows) {
                for (int j = 0; j < 8; ++j) vals[k][j] = 0.0f;
                continue;
            }
            const sycl::vec<sycl::half, 8> v = *reinterpret_cast<const sycl::vec<sycl::half, 8>*>(x + r * stride + tensor * HD + h * D + d);
            for (int j = 0; j < 8; ++j) {
                float f = (float) v[j];
                if (tensor == 1) f -= ksum[h * D + d + j] * inv_s;
                vals[k][j] = f;
                m = sycl::fmax(m, sycl::fabs(f));
            }
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
    if (tla.sage(&q, iq, ik, x + 2 * HD, out, qs, ks, (int) BLOCK, S, H, D, stride, o_rs, scale) != 0) return ns_fail(tla.err());
    return 0;
    NS_CATCH
}

}  // extern "C"
