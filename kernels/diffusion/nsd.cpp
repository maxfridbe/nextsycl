// nsd.cpp: the diffusion kernels the image and video engines share (text encoders, denoisers, VAEs, upscalers),
// linked into libnextsycl-image.so and libnextsycl-video.so. From MiniMax H3's SYCL engine (kernels/h3sycl.cpp at
// bd0f481, the user's h3sycl repository), its h3s_ functions renamed nsd_ and its settings H3S_ -> NSD_; the kernels
// unchanged. A context wraps a GPU's own queue (nsd_create(ns_gpu_queue(g))), so the buffers nextsycl allocates on
// that GPU are the kernels' to read.
// h3sycl.cpp - SYCL kernels for MiniMax H3 on Intel Arc (Xe2). The C ABI is h3sycl.h; two hosts use it:
//
//   the Rust engine (engine/)        nsd_open(): the library owns the queue and the device memory (nsd_alloc ...)
//   the PyTorch reference pipeline   nsd_create(queue): PyTorch's own in-order queue and its tensors' device pointers,
//                                    so calls are ordered with PyTorch's operations and nothing is copied
//
// Every data pointer is a USM device pointer of the context's queue.
//
// int8_linear, as comfy-kitchen's eager backend defines it:
//   x_rot = x . H           per group of `group` features, H the normalized regular Hadamard matrix (ConvRot)
//   s_r   = max|x_rot[r]| / 127                                  (per row, at least 1e-30)
//   q     = clamp(round(x_rot / s_r), -128, 127)                 int8
//   acc   = q . W^T                                              int32, W int8 [N, K]   <- the matrix engine, oneDNN
//   y     = cast(acc * (s_r * w_scale)) + bias                   in the output type
#include "nsd.h"

#include <sycl/sycl.hpp>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_graph.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>

#include <chrono>
#include <dlfcn.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace {

enum Dt : int { F32 = 0, F16 = 1, BF16 = 2 };

thread_local std::string g_err;

// bf16 <-> f32 on bits (SYCL has no bfloat16 type): widen = shift; narrow = round to nearest even
inline float bf16_to_f32(uint16_t b) { return sycl::bit_cast<float>((uint32_t) b << 16); }
inline uint16_t f32_to_bf16(float f) {
    uint32_t u = sycl::bit_cast<uint32_t>(f);
    if ((u & 0x7f800000u) == 0x7f800000u) return (uint16_t) (u >> 16);          // inf / nan: truncate
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}
inline float load(const void* p, int dt, size_t i) {
    switch (dt) {
        case F16: return (float) ((const sycl::half*) p)[i];
        case BF16: return bf16_to_f32(((const uint16_t*) p)[i]);
        default: return ((const float*) p)[i];
    }
}
inline void store(void* p, int dt, size_t i, float v) {
    switch (dt) {
        case F16: ((sycl::half*) p)[i] = (sycl::half) v; break;
        case BF16: ((uint16_t*) p)[i] = f32_to_bf16(v); break;
        default: ((float*) p)[i] = v;
    }
}
constexpr int kMaxGroup = 256;

// exp(x) for x <= 0, to about 2e-5 relative (the result is stored in a 16-bit float): 2^(x log2 e) with the
// fraction's power from a 6th-order polynomial and the integer part put straight into the exponent bits. The
// library's exp is several times slower, and attention needs one per query-key pair.
inline float exp_neg(float x) {
    const float t = sycl::fmax(x * 1.44269504f, -126.0f);
    const float n = sycl::floor(t);
    const float f = t - n;
    const float p = 1.0f + f * (0.693147182f + f * (0.240226507f + f * (0.0555041087f + f * (0.00961812911f + f * (0.00133335581f + f * 0.000154035304f)))));
    return p * sycl::bit_cast<float>((uint32_t) ((int32_t) n + 127) << 23);
}

struct Ctx {
    sycl::queue q;
    sycl::queue* cq = nullptr;                     // uploads beside the kernels (nsd_upload), made on first use
    std::mutex cq_mu;
    dnnl::engine eng;
    dnnl::stream strm;
    int8_t* xq = nullptr; size_t xq_cap = 0;       // quantized activations [M, K]
    float* rs = nullptr; size_t rs_cap = 0;        // per-row scale times the scalar weight scale [M]
    float* gmax = nullptr; size_t gmax_cap = 0;    // per (row, lane) partial absmax of the row-scale pass
    float* xr = nullptr; size_t xr_cap = 0;        // rotated activations [M, K], float32 (see pass 1)
    int32_t* acc = nullptr; size_t acc_cap = 0;    // fallback path: one chunk of int32 products [C, N]
    float* inv = nullptr; size_t inv_cap = 0;      // 1 / rms per row (norms)
    // attention, all in IEEE half: a chunk of q by head [H, rows, D]; k transposed [H, D, S]; v by head [H, S, D];
    // one chunk of scores [H, rows, S]; one chunk of results [H, rows, D]
    uint16_t* hq = nullptr; size_t hq_cap = 0;
    uint16_t* hk = nullptr; size_t hk_cap = 0;
    uint16_t* hv = nullptr; size_t hv_cap = 0;
    uint16_t* sc = nullptr; size_t sc_cap = 0;
    float* ao = nullptr; size_t ao_cap = 0;        // float32: the sums are over un-normalized weights
    float* part = nullptr; size_t part_cap = 0;    // per (row, lane): the max of that lane's share of the row
    float* rowv = nullptr; size_t rowv_cap = 0;    // per row: its max
    bool profile = false;                          // NSD_PROFILE: attention waits per phase and reports its timings
    struct Attn { dnnl::matmul qk; dnnl::matmul pv; };
    // oneDNN's fused attention, compiled per (rows, S, H, D); see attention_fused
    struct Sdpa {
        dnnl::graph::compiled_partition cp;
        std::vector<dnnl::graph::logical_tensor> in;    // in the partition's port order
        std::vector<int> slot;                          // which of (q, k, scale, v) each port takes
        dnnl::graph::logical_tensor out;
    };
    std::map<std::tuple<int64_t, int64_t, int64_t, int64_t>, Sdpa> sdpa;
    // nsd_attention_batch: (B, S, H, D, stride, in place) -> its compiled partition
    std::map<std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, bool>, Sdpa> sdpa_b;
    // nextsycl: prefix + block attention read in place through strides, by (rows, Skv, H, D, qs, kvs)
    std::map<std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>, Sdpa> sdpa_qk;
    bool sdpa_qk_direct_ok = true;
    bool sdpa_direct_ok = true;                    // false once oneDNN has refused the strided (copy-free) form
    float sdpa_scale = 0.0f;                       // 1 / sqrt(D): a host scalar oneDNN reads when the kernel runs
    bool sdpa_ok = true;                           // false once oneDNN has refused the fused form
    size_t attn_table_bytes = 1536ull << 20;       // the most score-table memory a chunk of attention may take
    // query rows per call of the fused kernel (NSD_ATTN_ROWS). At 47k keys: 1024 rows 749 ms a call, 2048 739,
    // 8192 719, all 47k 707 - and all of them is 2 GiB more of copies; 8192 is the trade.
    int64_t attn_rows = 8192;
    double lin_ms[4] = {0, 0, 0, 0};               // NSD_PROFILE: int8 linear time by pass (rotation, scales, quantize, GEMM)
    int64_t lin_calls = 0;
    std::map<std::tuple<int64_t, int64_t, int64_t, int64_t>, Attn> attn;                   // (H, rows, S, D)
    void* had[3] = {nullptr, nullptr, nullptr};    // the normalized 256 x 256 Hadamard matrix per Dt
    std::map<std::tuple<int64_t, int64_t, int64_t, int, int, int>, dnnl::matmul> gemm;   // (M, K, N, out, bias, per_n)
    std::map<std::tuple<int64_t, int, int>, dnnl::matmul> rot;                            // (rows, group, dt)
    std::map<std::tuple<int64_t, int64_t, int64_t, int, int>, dnnl::matmul> lin;          // (M, K, N, dt, bias)
    bool fused_ok = true;                          // oneDNN took the post-op form (else: int32 + our rescale kernel)
    bool rot_f32 = true;                           // oneDNN took a float32 result for the 16-bit rotation
    // the upscaler's 3-D convolutions: a primitive per (T, H, W, Ci, Co, k), the weights reordered once per buffer
    struct Conv3 { dnnl::convolution_forward prim; dnnl::convolution_forward::primitive_desc pd; };
    std::map<std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>, Conv3> conv3;
    std::map<const void*, dnnl::memory> conv3_w;
    // nsd_conv2d: (N, H, W, Ci, Co, k, dt) -> its primitive (the weights' reorders share conv3_w, keyed by pointer)
    std::map<std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int>, Conv3> conv2;
    std::map<std::tuple<std::vector<int64_t>>, Conv3> conv3x;   // strided, unpadded (nsd_conv3d_ex)
    float* gn = nullptr; size_t gn_cap = 0;        // group norm: partial sums, then (mean, 1 / std) per group
    bool rotq_fused = true;
    bool poison = false;
    // attention through libh3sage.so (kernels/sage.cpp), loaded on first use; NSD_ATTN=onednn: oneDNN's fused kernel
    bool sage_want = true;
    int64_t sage_min_s = 8192;                     // shorter sequences stay on oneDNN (NSD_SAGE_MIN_S): there the
                                                   // quantize pass and the call cost more than int8 saves
    int sage_state = 0;                            // 0 not loaded yet, 1 ready, -1 unavailable (oneDNN is used)
    int (*sage_fn)(void*, const int8_t*, const int8_t*, const void*, void*, const float*, const float*, int, int64_t,
                   int64_t, int64_t, float) = nullptr;
    const char* (*sage_err)() = nullptr;
    int8_t* sq = nullptr; size_t sq_cap = 0;       // q and k as int8 [H, S, D]
    int8_t* sk = nullptr; size_t sk_cap = 0;
    float* ss = nullptr; size_t ss_cap = 0;        // their scales, then k's mean [H, D]                           // NSD_POISON=1: new buffers filled with NaN bytes (finds reads of unwritten memory)                        // rotation + row scale + quantize in one kernel (NSD_ROTQ_SPLIT=1: the passes)
    // nsd_alloc's book: the xe driver has no out-of-memory error (an over-commit stalls the whole machine), so the
    // library refuses an allocation that would pass the cap instead of asking the driver
    std::mutex mem_mu;
    std::unordered_map<void*, size_t> mem;
    size_t mem_used = 0, mem_cap = 0;
    std::string name;

    // A scratch buffer of at least n elements; counted against the same cap as nsd_alloc. NULL (with the reason in
    // g_err) when it would pass the cap or the device refuses.
    template <typename T> T* grow(T*& p, size_t& cap, size_t n) {
        if (n <= cap) return p;
        const size_t want = n + (n >> 3);
        std::lock_guard<std::mutex> l(mem_mu);
        if (mem_used - cap * sizeof(T) + want * sizeof(T) > mem_cap) {
            g_err = "a scratch buffer of " + std::to_string((want * sizeof(T)) >> 20) + " MiB would pass the device memory cap (" +
                    std::to_string(mem_used >> 20) + " of " + std::to_string(mem_cap >> 20) + " MiB in use)";
            return nullptr;
        }
        if (p) { q.wait(); sycl::free(p, q); }
        mem_used -= cap * sizeof(T);
        p = sycl::malloc_device<T>(want, q);
        cap = p ? want : 0;
        mem_used += cap * sizeof(T);
        if (p && poison) q.memset(p, 0xff, want * sizeof(T)).wait();
        if (!p) g_err = "the device refused a scratch buffer of " + std::to_string((want * sizeof(T)) >> 20) + " MiB";
        return p;
    }
};

inline dnnl::memory::data_type ddt(int dt) {
    return dt == F16 ? dnnl::memory::data_type::f16 : dt == BF16 ? dnnl::memory::data_type::bf16
                                                                 : dnnl::memory::data_type::f32;
}
inline dnnl::memory usm(const dnnl::memory::desc& md, const dnnl::engine& eng, const void* p) {
    return dnnl::sycl_interop::make_memory(md, eng, dnnl::sycl_interop::memory_kind::usm, const_cast<void*>(p));
}

// the normalized regular Hadamard matrix of size g (h4 (x) h4 (x) ...), on the device in type dt
void* hadamard_matrix(Ctx& c, int g, int dt) {
    if (c.had[dt]) return c.had[dt];
    std::vector<float> h((size_t) g * g);
    static const int h4[4][4] = {{1, 1, 1, -1}, {1, 1, -1, 1}, {1, -1, 1, 1}, {-1, 1, 1, 1}};
    float norm = 1.0f;
    for (int s = 1; s < g; s *= 4) norm *= 0.5f;
    for (int i = 0; i < g; ++i)
        for (int j = 0; j < g; ++j) {
            int sign = 1;
            for (int a = i, b = j, s = 1; s < g; s *= 4, a /= 4, b /= 4) sign *= h4[a % 4][b % 4];
            h[(size_t) i * g + j] = (float) sign * norm;
        }
    const size_t n = (size_t) g * g;
    if (dt == F32) {
        float* d = sycl::malloc_device<float>(n, c.q);
        c.q.memcpy(d, h.data(), n * 4).wait();
        c.had[dt] = d;
    } else {
        std::vector<uint16_t> hb(n);
        for (size_t i = 0; i < n; ++i) {
            if (dt == BF16) hb[i] = f32_to_bf16(h[i]);
            else { sycl::half v = (sycl::half) h[i]; std::memcpy(&hb[i], &v, 2); }
        }
        uint16_t* d = sycl::malloc_device<uint16_t>(n, c.q);
        c.q.memcpy(d, hb.data(), n * 2).wait();
        c.had[dt] = d;
    }
    return c.had[dt];
}

// what both ways of making a context share: the oneDNN engine on the queue, the device's name, the memory cap
void init(Ctx& c) {
    c.eng = dnnl::sycl_interop::make_engine(c.q.get_device(), c.q.get_context());
    c.strm = dnnl::sycl_interop::make_stream(c.eng, c.q);
    c.name = c.q.get_device().get_info<sycl::info::device::name>();
    const uint64_t total = c.q.get_device().get_info<sycl::info::device::global_mem_size>();
    double frac = 0.94;                            // of the card, for nsd_alloc and the kernels' scratch together
    if (const char* e = std::getenv("NSD_MEM_FRACTION")) frac = std::min(0.97, std::max(0.05, std::atof(e)));
    c.mem_cap = (size_t) ((double) total * frac);
#ifdef NSD_SDPA_NO_FALLBACK
    // built against a oneDNN with container/onednn-sdpa-no-fallback.patch: ask it to refuse, not to fall back
    setenv("_ONEDNN_GRAPH_SDPA_NO_FALLBACK", "1", 1);
#endif
    c.profile = std::getenv("NSD_PROFILE") != nullptr;
    c.rotq_fused = std::getenv("NSD_ROTQ_SPLIT") == nullptr;
    c.poison = std::getenv("NSD_POISON") != nullptr;
    if (const char* e = std::getenv("NSD_ATTN")) c.sage_want = std::strcmp(e, "onednn") != 0;
    if (const char* e = std::getenv("NSD_SAGE_MIN_S")) c.sage_min_s = std::max<int64_t>(0, std::atoll(e));
    if (const char* e = std::getenv("NSD_ATTN_TABLE_MB")) c.attn_table_bytes = (size_t) std::max(64, std::atoi(e)) << 20;
    if (const char* e = std::getenv("NSD_ATTN_ROWS")) c.attn_rows = std::max(16, std::atoi(e));
}

}  // namespace

extern "C" {

const char* nsd_last_error(void) { return g_err.c_str(); }

void* nsd_create(void* sycl_queue) try {
    auto* c = new Ctx{*static_cast<sycl::queue*>(sycl_queue)};
    init(*c);
    return c;
} catch (const std::exception& e) { g_err = e.what(); return nullptr; }

void* nsd_open(void) try {
    auto* c = new Ctx{sycl::queue(sycl::gpu_selector_v, sycl::property::queue::in_order())};
    init(*c);
    return c;
} catch (const std::exception& e) { g_err = e.what(); return nullptr; }

// the GPUs, in the order the runtime lists them (with ONEAPI_DEVICE_SELECTOR=level_zero:* every card is there)
static std::vector<sycl::device> gpus() { return sycl::device::get_devices(sycl::info::device_type::gpu); }

int nsd_gpu_count(void) try {
    return (int) gpus().size();
} catch (const std::exception& e) { g_err = e.what(); return -1; }

int nsd_gpu_info(int index, char* name, int name_len, uint64_t* mem_bytes, char* pci, int pci_len) try {
    auto all = gpus();
    if (index < 0 || index >= (int) all.size()) { g_err = "no GPU " + std::to_string(index); return -1; }
    const auto& d = all[index];
    std::snprintf(name, name_len, "%s", d.get_info<sycl::info::device::name>().c_str());
    *mem_bytes = d.get_info<sycl::info::device::global_mem_size>();
    std::string addr = d.has(sycl::aspect::ext_intel_pci_address) ? d.get_info<sycl::ext::intel::info::device::pci_address>() : "";
    std::snprintf(pci, pci_len, "%s", addr.c_str());
    return 0;
} catch (const std::exception& e) { g_err = e.what(); return -1; }

void* nsd_open_gpu(int index) try {
    auto all = gpus();
    if (index < 0 || index >= (int) all.size()) { g_err = "no GPU " + std::to_string(index) + " (" + std::to_string(all.size()) + " found)"; return nullptr; }
    // a context of this device alone: in the platform's default context (every GPU) Level Zero makes each device
    // allocation reachable from the other cards too, through system-RAM-backed GTT mappings - 2026-10-05 a B65
    // worker held its 19.9 GB model in its VRAM AND 19.8 GB of host RAM mapped to the B70, and two workers with the
    // text encoder's pinned copies ran the host out of memory (the OOM killer took the user's session processes)
    sycl::context own(all[index]);
    auto* c = new Ctx{sycl::queue(own, all[index], sycl::property::queue::in_order())};
    init(*c);
    return c;
} catch (const std::exception& e) { g_err = e.what(); return nullptr; }

const char* nsd_device_name(void* ctx) { return static_cast<Ctx*>(ctx)->name.c_str(); }
uint64_t nsd_mem_cap(void* ctx) { return static_cast<Ctx*>(ctx)->mem_cap; }
uint64_t nsd_mem_free(void* ctx) try {
    const auto dev = static_cast<Ctx*>(ctx)->q.get_device();
    if (!dev.has(sycl::aspect::ext_intel_free_memory)) return 0;
    return dev.get_info<sycl::ext::intel::info::device::free_memory>();
} catch (const std::exception&) { return 0; }
uint64_t nsd_mem_used(void* ctx) {
    auto& c = *static_cast<Ctx*>(ctx);
    std::lock_guard<std::mutex> l(c.mem_mu);
    return c.mem_used;
}

void* nsd_alloc(void* ctx, uint64_t bytes) try {
    auto& c = *static_cast<Ctx*>(ctx);
    std::lock_guard<std::mutex> l(c.mem_mu);
    if (c.mem_used + bytes > c.mem_cap) {
        g_err = "nsd_alloc: " + std::to_string(bytes >> 20) + " MiB would pass the device memory cap (" +
                std::to_string(c.mem_used >> 20) + " of " + std::to_string(c.mem_cap >> 20) + " MiB in use)";
        return nullptr;
    }
    void* p = sycl::malloc_device(bytes ? bytes : 1, c.q);
    if (!p) { g_err = "nsd_alloc: the device refused " + std::to_string(bytes >> 20) + " MiB"; return nullptr; }
    if (c.poison) c.q.memset(p, 0xff, bytes ? bytes : 1).wait();
    c.mem.emplace(p, bytes);
    c.mem_used += bytes;
    return p;
} catch (const std::exception& e) { g_err = e.what(); return nullptr; }

void nsd_free(void* ctx, void* p) {
    auto& c = *static_cast<Ctx*>(ctx);
    if (!p) return;
    c.q.wait();                                    // nothing queued may still read or write it
    if (c.cq) c.cq->wait();                        // nor an upload into it
    std::lock_guard<std::mutex> l(c.mem_mu);
    auto it = c.mem.find(p);
    if (it == c.mem.end()) return;
    // weights reordered for oneDNN are cached by their buffer's address, and the allocator hands a freed address
    // out again: a later load (the next clip's video encoder after this one's upscaler) would find this buffer's
    // reorder and convolve with another network's weights
    const char* lo = static_cast<const char*>(p);
    for (auto w = c.conv3_w.lower_bound(lo); w != c.conv3_w.end() && static_cast<const char*>(w->first) < lo + it->second;)
        w = c.conv3_w.erase(w);
    c.mem_used -= it->second;
    c.mem.erase(it);
    sycl::free(p, c.q);
}

int nsd_write(void* ctx, void* dst, const void* src_host, uint64_t bytes) try {
    static_cast<Ctx*>(ctx)->q.memcpy(dst, src_host, bytes).wait();
    return 0;
} catch (const std::exception& e) { g_err = e.what(); return -1; }

int nsd_read(void* ctx, void* dst_host, const void* src, uint64_t bytes) try {
    static_cast<Ctx*>(ctx)->q.memcpy(dst_host, src, bytes).wait();
    return 0;
} catch (const std::exception& e) { g_err = e.what(); return -1; }

int nsd_copy(void* ctx, void* dst, const void* src, uint64_t bytes) try {
    static_cast<Ctx*>(ctx)->q.memcpy(dst, src, bytes);
    return 0;
} catch (const std::exception& e) { g_err = e.what(); return -1; }

int nsd_wait(void* ctx) try {
    static_cast<Ctx*>(ctx)->q.wait_and_throw();
    return 0;
} catch (const std::exception& e) { g_err = e.what(); return -1; }

void* nsd_alloc_host(void* ctx, uint64_t bytes) try {
    auto& c = *static_cast<Ctx*>(ctx);
    void* p = sycl::malloc_host(bytes ? bytes : 1, c.q);
    if (!p) g_err = "nsd_alloc_host: the runtime refused " + std::to_string(bytes >> 20) + " MiB of pinned memory";
    return p;
} catch (const std::exception& e) { g_err = e.what(); return nullptr; }

void nsd_free_host(void* ctx, void* p) {
    auto& c = *static_cast<Ctx*>(ctx);
    if (!p) return;
    if (c.cq) c.cq->wait();
    sycl::free(p, c.q);
}

int nsd_upload(void* ctx, void* dst, const void* src_pinned, uint64_t bytes) try {
    auto& c = *static_cast<Ctx*>(ctx);
    const auto ctx_ = c.q.get_context();
    if (sycl::get_pointer_type(src_pinned, ctx_) != sycl::usm::alloc::host || sycl::get_pointer_type(dst, ctx_) != sycl::usm::alloc::device) {
        g_err = "nsd_upload: the source must be nsd_alloc_host memory and the destination device memory";
        return -1;
    }
    {
        std::lock_guard<std::mutex> l(c.cq_mu);
        if (!c.cq) c.cq = new sycl::queue(ctx_, c.q.get_device(), sycl::property::queue::in_order());
    }
    c.cq->memcpy(dst, src_pinned, bytes);
    return 0;
} catch (const std::exception& e) { g_err = e.what(); return -1; }

int nsd_upload_wait(void* ctx) try {
    auto& c = *static_cast<Ctx*>(ctx);
    if (c.cq) c.cq->wait_and_throw();
    return 0;
} catch (const std::exception& e) { g_err = e.what(); return -1; }

void nsd_destroy(void* ctx) {
    auto* c = static_cast<Ctx*>(ctx);
    if (!c) return;
    c->q.wait();
    if (c->cq) { c->cq->wait(); delete c->cq; c->cq = nullptr; }
    if (c->profile && c->lin_calls)
        std::fprintf(stderr, "h3sycl: int8 linear, %lld row chunks: rotation %.1f ms, row scales %.1f, quantize %.1f, int8 GEMM %.1f\n",
                     (long long) c->lin_calls, c->lin_ms[0], c->lin_ms[1], c->lin_ms[2], c->lin_ms[3]);
    for (void* p : {(void*) c->xq, (void*) c->rs, (void*) c->gmax, (void*) c->xr, (void*) c->acc, c->had[0], c->had[1], c->had[2],
                    (void*) c->inv, (void*) c->hq, (void*) c->hk, (void*) c->hv, (void*) c->sc, (void*) c->ao, (void*) c->part, (void*) c->rowv})
        if (p) sycl::free(p, c->q);
    for (auto& [p, n] : c->mem) sycl::free(p, c->q);
    delete c;
}

// Rotation, row scale and quantization in one kernel, for groups of 256 (the int8_convrot checkpoints): reads the
// 16-bit input once and writes the int8 copy and the row scales, where the passes below write and read a float32
// copy of the input three times.
//
// One work-group per row, 16 sub-groups of 16 lanes; a sub-group rotates one group of 256 features at a time, lane
// L holding features L*16 .. L*16+15 of it. The normalized Hadamard matrix of 256 is h4 (x) h4 (x) h4 (x) h4, one
// factor per base-4 digit of the feature index, so it applies digit by digit in any order: the two low digits
// inside each lane's registers, the two high digits across the lanes (shuffles). For one digit, with x_0..x_3 the
// four values that differ only in that digit, h4 gives y_a = (x_0 + x_1 + x_2 + x_3) - 2 x_(3-a).
// KG: the most groups one sub-group holds (in registers) for a row; K <= KG * 16 * 256.
extern "C++" {
template <int KG>
static void rotate_quantize(sycl::queue& q, const void* x, int x_dt, int64_t M, int64_t K, int8_t* xq, float* rs) {
    constexpr int kSg = 16, kNsg = 16;
    const int64_t G = K / 256;
    q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) M * kSg * kNsg), sycl::range<1>(kSg * kNsg)),
                   [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        const auto sg = it.get_sub_group();
        const int lane = (int) sg.get_local_id()[0], s = (int) sg.get_group_id()[0];
        const size_t r = it.get_group(0);
        float v[KG][16];
        float m = 0.0f;
#pragma unroll
        for (int k = 0; k < KG; ++k) {
            const int64_t gi = s + (int64_t) k * kNsg;
            if (gi >= G) break;     // the same for every lane of the sub-group
            const size_t base = r * K + gi * 256 + lane * 16;
#pragma unroll
            for (int e = 0; e < 16; ++e) v[k][e] = load(x, x_dt, base + e);
            // low two digits: inside the lane
#pragma unroll
            for (int d = 1; d <= 4; d *= 4) {
#pragma unroll
                for (int e0 = 0; e0 < 16; ++e0) {
                    if ((e0 / d) % 4 != 0) continue;
                    const float a0 = v[k][e0], a1 = v[k][e0 + d], a2 = v[k][e0 + 2 * d], a3 = v[k][e0 + 3 * d];
                    const float t = a0 + a1 + a2 + a3;
                    v[k][e0] = t - 2.0f * a3; v[k][e0 + d] = t - 2.0f * a2;
                    v[k][e0 + 2 * d] = t - 2.0f * a1; v[k][e0 + 3 * d] = t - 2.0f * a0;
                }
            }
            // high two digits: across the lanes
#pragma unroll
            for (int e = 0; e < 16; ++e) {
                float a = v[k][e];
#pragma unroll
                for (int sh = 0; sh <= 2; sh += 2) {
                    const float p1 = sycl::permute_group_by_xor(sg, a, 1 << sh);
                    const float t2 = a + p1;
                    const float t = t2 + sycl::permute_group_by_xor(sg, t2, 2 << sh);
                    a = t - 2.0f * sycl::permute_group_by_xor(sg, a, 3 << sh);
                }
                a *= 1.0f / 16.0f;
                v[k][e] = a;
                m = sycl::fmax(m, sycl::fabs(a));
            }
        }
        m = sycl::reduce_over_group(it.get_group(), m, sycl::maximum<float>());
        const float scale = sycl::fmax(m / 127.0f, 1e-30f);
        if (it.get_local_id(0) == 0) rs[r] = scale;
#pragma unroll
        for (int k = 0; k < KG; ++k) {
            const int64_t gi = s + (int64_t) k * kNsg;
            if (gi >= G) break;
            const size_t base = r * K + gi * 256 + lane * 16;
#pragma unroll
            for (int e = 0; e < 16; ++e) xq[base + e] = (int8_t) sycl::clamp(sycl::rint(v[k][e] / scale), -128.0f, 127.0f);
        }
    });
}
}  // extern "C++"

// The passes (the contract is in h3sycl.h):
//   1. rotation     x_rot = x . H per group: ONE GEMM [M * K / g, g] x [g, g] on the matrix engine (it is ~0.3 ms;
//                   a scalar butterfly kernel was 5 ms a pass)
//   2. row scales   absmax per (row, group), then per row
//   3. quantize     one value per work-item
//   4. int8 GEMM    oneDNN, the rescale (per-row scale, optional per-column weight scale) and the bias as post-ops,
//                   written straight into the output in its own type - no int32 table, no second pass over it
// M rows of the layer (nsd_int8_linear below hands it the rows a chunk at a time).
static int int8_linear_rows(Ctx& c, const void* x, int x_dt, int64_t M, int64_t K, const int8_t* w, int64_t N,
                            const float* wscale, int64_t n_wscale, const float* bias, void* out, int out_dt, int group) {
    const bool rot = group > 0;
    const int g = rot ? group : 256;
    // the row-max pass: one value per work-item and a work-group reduction to one max per 256 features, then a
    // short pass per row over those. At 47k tokens this pass, the rotation before it and the quantize after it take
    // ~110 ms a block between them, against ~190 for the int8 GEMM: three passes over a float32 copy of the input.
    constexpr int64_t kWg = 256;
    const int64_t G = (K + kWg - 1) / kWg;
    int8_t* xq = c.grow(c.xq, c.xq_cap, (size_t) M * K);
    float* rs = c.grow(c.rs, c.rs_cap, (size_t) M);
    float* gmax = c.grow(c.gmax, c.gmax_cap, (size_t) M * G);
    if (!xq || !rs || !gmax) return -1;
    sycl::queue& q = c.q;
    using dnnl::memory;
    auto clock = std::chrono::steady_clock::now();
    auto lap = [&](int i) {
        if (!c.profile) return;
        q.wait();
        const auto now = std::chrono::steady_clock::now();
        c.lin_ms[i] += std::chrono::duration<double, std::milli>(now - clock).count();
        clock = now;
    };
    if (c.profile) { q.wait(); clock = std::chrono::steady_clock::now(); ++c.lin_calls; }

    const bool per_n = n_wscale > 1;
    const int64_t kg = (K / 256 + 15) / 16;
    if (rot && g == 256 && c.rotq_fused && kg <= 4) {
        // ---- 1-3 in one kernel
        switch (kg) {
            case 1: rotate_quantize<1>(q, x, x_dt, M, K, xq, rs); break;
            case 2: rotate_quantize<2>(q, x, x_dt, M, K, xq, rs); break;
            case 3: rotate_quantize<3>(q, x, x_dt, M, K, xq, rs); break;
            default: rotate_quantize<4>(q, x, x_dt, M, K, xq, rs); break;
        }
        lap(0);
    } else {
    // ---- 1. rotation on the matrix engine
    const void* xs = x;          // what the quantizer reads
    int xs_dt = x_dt;
    if (rot) {
        float* xr = c.grow(c.xr, c.xr_cap, (size_t) M * K);
        if (!xr) return -1;
        // The rotated copy is float32. In bfloat16 (8 significant bits) its rounding moves ~5% of the values to the
        // neighbouring int8 level in pass 3: 0.7% error on the layer's output, measured against exact arithmetic,
        // next to the 0.9% the 8-bit quantization costs by itself.
        int r_dt = c.rot_f32 ? F32 : x_dt;
        const int64_t rows = M * (K / g);
        memory::desc smd({rows, g}, ddt(x_dt), memory::format_tag::ab);
        memory::desc hmd({g, g}, ddt(x_dt), memory::format_tag::ab);
        auto key = std::make_tuple(rows, g, x_dt);
        auto it = c.rot.find(key);
        if (it == c.rot.end()) {
            try {
                memory::desc d32({rows, g}, ddt(r_dt), memory::format_tag::ab);
                it = c.rot.emplace(key, dnnl::matmul(dnnl::matmul::primitive_desc(c.eng, smd, hmd, d32))).first;
            } catch (const dnnl::error& e) {
                if (r_dt == x_dt) throw;
                std::fprintf(stderr, "h3sycl: oneDNN refused a float32 rotation result (%s); keeping the input's type\n", e.what());
                c.rot_f32 = false; r_dt = x_dt;
                memory::desc d16({rows, g}, ddt(r_dt), memory::format_tag::ab);
                it = c.rot.emplace(key, dnnl::matmul(dnnl::matmul::primitive_desc(c.eng, smd, hmd, d16))).first;
            }
        }
        memory::desc dmd({rows, g}, ddt(r_dt), memory::format_tag::ab);
        it->second.execute(c.strm, {{DNNL_ARG_SRC, usm(smd, c.eng, x)}, {DNNL_ARG_WEIGHTS, usm(hmd, c.eng, hadamard_matrix(c, g, x_dt))},
                                    {DNNL_ARG_DST, usm(dmd, c.eng, xr)}});
        xs = xr; xs_dt = r_dt;
    }
    lap(0);
    // ---- 2. row scales
    q.parallel_for(sycl::nd_range<2>(sycl::range<2>((size_t) M, (size_t) (G * kWg)), sycl::range<2>(1, (size_t) kWg)),
                   [=](sycl::nd_item<2> it) {
        const size_t r = it.get_global_id(0), i = it.get_global_id(1);
        const float v = i < (size_t) K ? sycl::fabs(load(xs, xs_dt, r * K + i)) : 0.0f;
        const float m = sycl::reduce_over_group(it.get_group(), v, sycl::maximum<float>());
        if (it.get_local_id(1) == 0) gmax[r * G + it.get_group(1)] = m;
    });
    q.parallel_for(sycl::range<1>((size_t) M), [=](sycl::id<1> r) {
        float m = 0.0f;
        for (int64_t j = 0; j < G; ++j) m = sycl::fmax(m, gmax[r[0] * G + j]);
        // the row's quantization step; times the scalar weight scale it is the whole rescale factor
        rs[r[0]] = sycl::fmax(m / 127.0f, 1e-30f);
    });
    lap(1);
    // ---- 3. quantize
    q.parallel_for(sycl::range<2>((size_t) M, (size_t) K), [=](sycl::id<2> id) {
        const size_t i = id[0] * K + id[1];
        xq[i] = (int8_t) sycl::clamp(sycl::rint(load(xs, xs_dt, i) / rs[id[0]]), -128.0f, 127.0f);
    });
    lap(2);
    }
    if (!per_n) q.parallel_for(sycl::range<1>((size_t) M), [=](sycl::id<1> r) { rs[r[0]] *= wscale[0]; });
    // ---- 4. int8 GEMM with the rescale and the bias as post-ops
    memory::desc smd({M, K}, memory::data_type::s8, memory::format_tag::ab);
    memory::desc wmd({K, N}, memory::data_type::s8, memory::format_tag::ba);     // the [N, K] buffer, read transposed
    memory::desc row_md({M, 1}, memory::data_type::f32, memory::format_tag::ab);
    memory::desc col_md({1, N}, memory::data_type::f32, memory::format_tag::ab);
    if (c.fused_ok) {
        try {
            memory::desc dmd({M, N}, ddt(out_dt), memory::format_tag::ab);
            auto key = std::make_tuple(M, K, N, out_dt, bias ? 1 : 0, per_n ? 1 : 0);
            auto it = c.gemm.find(key);
            if (it == c.gemm.end()) {
                dnnl::post_ops po;
                po.append_binary(dnnl::algorithm::binary_mul, row_md);
                if (per_n) po.append_binary(dnnl::algorithm::binary_mul, col_md);
                if (bias) po.append_binary(dnnl::algorithm::binary_add, col_md);
                dnnl::primitive_attr attr;
                attr.set_post_ops(po);
                it = c.gemm.emplace(key, dnnl::matmul(dnnl::matmul::primitive_desc(c.eng, smd, wmd, dmd, attr))).first;
            }
            std::unordered_map<int, memory> args{{DNNL_ARG_SRC, usm(smd, c.eng, xq)}, {DNNL_ARG_WEIGHTS, usm(wmd, c.eng, w)},
                                                 {DNNL_ARG_DST, usm(dmd, c.eng, out)}};
            int po_i = 0;
            args.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP(po_i++) | DNNL_ARG_SRC_1, usm(row_md, c.eng, rs)});
            if (per_n) args.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP(po_i++) | DNNL_ARG_SRC_1, usm(col_md, c.eng, wscale)});
            if (bias) args.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP(po_i++) | DNNL_ARG_SRC_1, usm(col_md, c.eng, bias)});
            it->second.execute(c.strm, args);
            lap(3);
            return 0;
        } catch (const dnnl::error& e) {
            std::fprintf(stderr, "h3sycl: oneDNN refused the fused int8 GEMM (%s); using the int32 + rescale path\n", e.what());
            c.fused_ok = false;
        }
    }
    // ---- fallback: int32 products in row chunks, rescaled by a kernel of ours
    const int64_t C = std::max<int64_t>(1, std::min<int64_t>(M, (int64_t) (384ull << 20) / (N * 4)));
    int32_t* acc = c.grow(c.acc, c.acc_cap, (size_t) C * N);
    if (!acc) return -1;
    for (int64_t r0 = 0; r0 < M; r0 += C) {
        const int64_t rows = std::min(C, M - r0);
        memory::desc s2({rows, K}, memory::data_type::s8, memory::format_tag::ab);
        memory::desc d2({rows, N}, memory::data_type::s32, memory::format_tag::ab);
        auto key = std::make_tuple(rows, K, N, -1, 0, 0);
        auto it = c.gemm.find(key);
        if (it == c.gemm.end()) it = c.gemm.emplace(key, dnnl::matmul(dnnl::matmul::primitive_desc(c.eng, s2, wmd, d2))).first;
        it->second.execute(c.strm, {{DNNL_ARG_SRC, usm(s2, c.eng, xq + r0 * K)}, {DNNL_ARG_WEIGHTS, usm(wmd, c.eng, w)},
                                    {DNNL_ARG_DST, usm(d2, c.eng, acc)}});
        q.parallel_for(sycl::range<2>((size_t) rows, (size_t) N), [=](sycl::id<2> id) {
            const size_t r = id[0], n = id[1];
            float v = (float) acc[r * N + n] * rs[r0 + r] * (per_n ? wscale[n] : 1.0f);
            if (bias) v += bias[n];
            store(out, out_dt, (size_t) (r0 + r) * N + n, v);
        });
    }
    return 0;
}

int nsd_int8_linear(void* ctx, const void* x, int x_dt, int64_t M, int64_t K, const int8_t* w, int64_t N,
                    const float* wscale, int64_t n_wscale, const float* bias, void* out, int out_dt, int group) try {
    auto& c = *static_cast<Ctx*>(ctx);
    if (M <= 0 || K <= 0 || N <= 0) return 0;
    if (group > 0 && (group > kMaxGroup || K % group != 0 || (group & (group - 1)) != 0)) {
        g_err = "nsd_int8_linear: unsupported ConvRot group size"; return -1;
    }
    // Row chunks: the rotated float32 copy and the int8 copy of the activations are scratch, [rows, K] each. For a
    // whole 47k-token input of the MLP's second linear that is 2.7 + 0.7 GiB; in chunks of 8192 rows, 0.6 GiB. The
    // matrix engine's rate does not depend on the number of rows.
    constexpr int64_t kRows = 8192;
    const size_t xe = x_dt == F32 ? 4 : 2, oe = out_dt == F32 ? 4 : 2;
    for (int64_t r0 = 0; r0 < M; r0 += kRows) {
        const int64_t rows = std::min(kRows, M - r0);
        const int rc = int8_linear_rows(c, (const char*) x + (size_t) r0 * K * xe, x_dt, rows, K, w, N, wscale, n_wscale, bias,
                                        (char*) out + (size_t) r0 * N * oe, out_dt, group);
        if (rc != 0) return rc;
    }
    return 0;
} catch (const std::exception& e) { g_err = e.what(); return -1; }

int nsd_linear(void* ctx, const void* x, int dt, int64_t M, int64_t K, const void* w, int64_t N, const float* bias,
               void* out, int out_dt) try {
    auto& c = *static_cast<Ctx*>(ctx);
    if (M <= 0 || K <= 0 || N <= 0) return 0;
    using dnnl::memory;
    constexpr int64_t kRows = 8192;                 // row chunks, so the scratch of a type change stays small
    const size_t xe = dt == F32 ? 4 : 2, oe = out_dt == F32 ? 4 : 2;
    const bool convert = out_dt != dt;              // oneDNN multiplies within one type; another output type is a pass of ours
    void* tmp = nullptr;
    if (convert) {
        tmp = c.grow(c.xr, c.xr_cap, (size_t) std::min(kRows, M) * N);      // float32-sized slots: enough for any type
        if (!tmp) return -1;
    }
    memory::desc wmd({K, N}, ddt(dt), memory::format_tag::ba);              // the [N, K] buffer, read transposed
    memory::desc col_md({1, N}, memory::data_type::f32, memory::format_tag::ab);
    for (int64_t r0 = 0; r0 < M; r0 += kRows) {
        const int64_t rows = std::min(kRows, M - r0);
        memory::desc smd({rows, K}, ddt(dt), memory::format_tag::ab);
        memory::desc dmd({rows, N}, ddt(dt), memory::format_tag::ab);
        auto key = std::make_tuple(rows, K, N, dt, bias ? 1 : 0);
        auto it = c.lin.find(key);
        if (it == c.lin.end()) {
            dnnl::primitive_attr attr;
            if (bias) {
                dnnl::post_ops po;
                po.append_binary(dnnl::algorithm::binary_add, col_md);
                attr.set_post_ops(po);
            }
            it = c.lin.emplace(key, dnnl::matmul(dnnl::matmul::primitive_desc(c.eng, smd, wmd, dmd, attr))).first;
        }
        void* dst = convert ? tmp : (char*) out + (size_t) r0 * N * oe;
        std::unordered_map<int, memory> args{{DNNL_ARG_SRC, usm(smd, c.eng, (const char*) x + (size_t) r0 * K * xe)},
                                             {DNNL_ARG_WEIGHTS, usm(wmd, c.eng, w)}, {DNNL_ARG_DST, usm(dmd, c.eng, dst)}};
        if (bias) args.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, usm(col_md, c.eng, bias)});
        it->second.execute(c.strm, args);
        if (convert) {
            const size_t base = (size_t) r0 * N;
            c.q.parallel_for(sycl::range<1>((size_t) (rows * N)), [=](sycl::id<1> i) {
                store(out, out_dt, base + i[0], load(tmp, dt, i[0]));
            });
        }
    }
    return 0;
} catch (const std::exception& e) { g_err = e.what(); return -1; }

// out += x . W^T, in the inputs' type (a LoRA's second factor, its strength folded into W): oneDNN's sum post-op,
// in row chunks like nsd_linear.
int nsd_linear_acc(void* ctx, const void* x, int dt, int64_t M, int64_t K, const void* w, int64_t N, const float* bias, void* out) try {
    auto& c = *static_cast<Ctx*>(ctx);
    if (M <= 0 || K <= 0 || N <= 0) return 0;
    using dnnl::memory;
    constexpr int64_t kRows = 8192;
    const size_t xe = dt == F32 ? 4 : 2;
    memory::desc wmd({K, N}, ddt(dt), memory::format_tag::ba);
    for (int64_t r0 = 0; r0 < M; r0 += kRows) {
        const int64_t rows = std::min(kRows, M - r0);
        memory::desc smd({rows, K}, ddt(dt), memory::format_tag::ab);
        memory::desc dmd({rows, N}, ddt(dt), memory::format_tag::ab);
        auto key = std::make_tuple(rows, K, N, dt, bias ? 3 : 2);
        auto it = c.lin.find(key);
        if (it == c.lin.end()) {
            dnnl::post_ops po;
            if (bias) po.append_binary(dnnl::algorithm::binary_add, memory::desc({1, N}, memory::data_type::f32, memory::format_tag::ab));
            po.append_sum(1.0f);
            dnnl::primitive_attr attr;
            attr.set_post_ops(po);
            it = c.lin.emplace(key, dnnl::matmul(dnnl::matmul::primitive_desc(c.eng, smd, wmd, dmd, attr))).first;
        }
        std::unordered_map<int, memory> args{{DNNL_ARG_SRC, usm(smd, c.eng, (const char*) x + (size_t) r0 * K * xe)}, {DNNL_ARG_WEIGHTS, usm(wmd, c.eng, w)},
                                             {DNNL_ARG_DST, usm(dmd, c.eng, (char*) out + (size_t) r0 * N * xe)}};
        if (bias) args.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, usm(memory::desc({1, N}, memory::data_type::f32, memory::format_tag::ab), c.eng, bias)});
        it->second.execute(c.strm, args);
    }
    return 0;
} catch (const std::exception& e) { g_err = e.what(); return -1; }

// rows of x [M, C] scaled in place, row r by s[r] (folds a per-output scale into a weight matrix)
int nsd_scale_rows(void* ctx, void* x, int dt, int64_t M, int64_t C, const float* s) try {
    auto& c = *static_cast<Ctx*>(ctx);
    if (M <= 0 || C <= 0) return 0;
    c.q.parallel_for(sycl::range<2>((size_t) M, (size_t) C), [=](sycl::id<2> id) {
        const size_t i = id[0] * C + id[1];
        store(x, dt, i, load(x, dt, i) * s[id[0]]);
    });
    return 0;
} catch (const std::exception& e) { g_err = e.what(); return -1; }

extern "C++" {
// rms_norm_mod's per-element pass for 16-bit x and out, 8 features per work-item (see nsd_rms_norm_mod)
template <int DT>
static void norm16(sycl::queue& q, const uint16_t* x, int64_t M, int64_t C, const float* inv, const float* weight,
                   const int32_t* rows, const float* scale, const float* shift, uint16_t* out) {
    q.parallel_for(sycl::range<2>((size_t) M, (size_t) C / 8), [=](sycl::id<2> id) {
        const size_t r = id[0], j = id[1] * 8;
        using V = sycl::vec<uint16_t, 8>;
        const V xv = *reinterpret_cast<const V*>(x + r * C + j);
        V ov;
        for (int e = 0; e < 8; ++e) {
            const size_t i = j + e;
            const uint16_t b = xv[e];
            float v = (DT == F16 ? (float) sycl::bit_cast<sycl::half>(b) : bf16_to_f32(b)) * inv[r] * weight[i];
            if (rows) {
                const size_t m = (size_t) rows[r] * C + i;
                v = v * (1.0f + scale[m]) + shift[m];
            }
            ov[e] = DT == F16 ? sycl::bit_cast<uint16_t>((sycl::half) v) : f32_to_bf16(v);
        }
        *reinterpret_cast<V*>(out + r * C + j) = ov;
    });
}
}  // extern "C++"

// 1 / sqrt(mean(x[r]^2) + eps) per row of x [M, C] into c.inv: a work-group of 256 per row, summed in float32
static float* row_inv_rms(Ctx& c, const void* x, int x_dt, int64_t M, int64_t C, float eps) {
    float* inv = c.grow(c.inv, c.inv_cap, (size_t) M);
    if (!inv) return nullptr;
    constexpr int64_t kWg = 256;
    c.q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (M * kWg)), sycl::range<1>(kWg)), [=](sycl::nd_item<1> it) {
        const int64_t r = it.get_group(0), lane = it.get_local_id(0);
        float s = 0.0f;
        for (int64_t i = lane; i < C; i += kWg) {
            const float v = load(x, x_dt, (size_t) (r * C + i));
            s += v * v;
        }
        s = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>());
        if (lane == 0) inv[r] = sycl::rsqrt(s / (float) C + eps);
    });
    return inv;
}

int nsd_rms_norm_mod(void* ctx, const void* x, int x_dt, int64_t M, int64_t C, const float* weight, float eps,
                     const int32_t* rows, const float* scale, const float* shift, void* out, int out_dt) try {
    auto& c = *static_cast<Ctx*>(ctx);
    if (M <= 0 || C <= 0) return 0;
    const float* inv = row_inv_rms(c, x, x_dt, M, C, eps);
    if (!inv) return -1;
    const bool mod = rows && scale && shift;
    // 16-bit in and out: 8 features per work-item through 16-byte loads, the same arithmetic per element (identical
    // results); the element-per-work-item form below ran well under the memory's rate
    const bool vec = x_dt == out_dt && (x_dt == F16 || x_dt == BF16) && C % 8 == 0 && (uintptr_t) x % 16 == 0 &&
                     (uintptr_t) out % 16 == 0 && !std::getenv("NSD_NORM_SCALAR");
    if (vec) {
        if (x_dt == F16) norm16<F16>(c.q, (const uint16_t*) x, M, C, inv, weight, mod ? rows : nullptr, scale, shift, (uint16_t*) out);
        else norm16<BF16>(c.q, (const uint16_t*) x, M, C, inv, weight, mod ? rows : nullptr, scale, shift, (uint16_t*) out);
        return 0;
    }
    c.q.parallel_for(sycl::range<2>((size_t) M, (size_t) C), [=](sycl::id<2> id) {
        const size_t r = id[0], i = id[1];
        float v = load(x, x_dt, r * C + i) * inv[r] * weight[i];
        if (mod) {
            const size_t m = (size_t) rows[r] * C + i;
            v = v * (1.0f + scale[m]) + shift[m];
        }
        store(out, out_dt, r * C + i, v);
    });
    return 0;
} catch (const std::exception& e) { g_err = e.what(); return -1; }

// ---- the text encoder (Qwen3, GGUF weights)

// llama.cpp's k-quant blocks of 256 values -> a 16-bit type. Q4_K: f16 d, f16 dmin, 12 bytes of 6-bit scales and
// mins for 8 sub-blocks of 32, 128 bytes of 4-bit values (144 bytes). Q6_K: 128 bytes low 4 bits, 64 bytes high 2
// bits, 16 int8 scales (one per 16 values), f16 d (210 bytes). One work-item per 32 values.
int nsd_dequant(void* ctx, const void* src, int qtype, int64_t n, void* out, int out_dt) try {
    auto& c = *static_cast<Ctx*>(ctx);
    if (n <= 0) return 0;
    const uint8_t* b = static_cast<const uint8_t*>(src);
    auto h2f = [](const uint8_t* p) { sycl::half h; std::memcpy(&h, p, 2); return (float) h; };
    // nextsycl: Q8_0 (blocks of 32: an fp16 scale, 32 int8) and BF16 (30), for image and video weights stored so
    if (qtype == 8) {
        if (n % 32 != 0) { g_err = "nsd_dequant: Q8_0 is a whole number of 32-value blocks"; return -1; }
        c.q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
            const size_t i = id[0];
            const uint8_t* x = b + (i / 32) * 34;
            store(out, out_dt, i, h2f(x) * (float) (int8_t) x[2 + i % 32]);
        });
        return 0;
    }
    if (qtype == 30) {
        c.q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
            const size_t i = id[0];
            store(out, out_dt, i, bf16_to_f32((uint16_t) (b[2 * i] | (b[2 * i + 1] << 8))));
        });
        return 0;
    }
    if (n % 256 != 0) { g_err = "nsd_dequant: a whole number of 256-value blocks"; return -1; }
    // four consecutive values per work-item, so a sub-group writes one contiguous run (2026-10-05: one work-item per
    // 32-value run wrote 32 strided streams, ~60 ms of a 16k-token block's 156 ms of Q4_K matrices on the B65); the
    // arithmetic per value is the same as before, so are the results
    const size_t items = (size_t) (n / 4);
    if (qtype == 12) {   // Q4_K
        c.q.parallel_for(sycl::range<1>(items), [=](sycl::id<1> id) {
            const size_t v0 = id[0] * 4;
            const int64_t blk = (int64_t) (v0 / 256);
            const int j0 = (int) (v0 % 256), j = j0 / 32, l0 = j0 % 32;     // sub-block j of 32, position l0 in it
            const uint8_t* x = b + blk * 144;
            const float d = h2f(x), dmin = h2f(x + 2);
            const uint8_t* sc = x + 4;
            int s, m;
            if (j < 4) { s = sc[j] & 63; m = sc[j + 4] & 63; }
            else { s = (sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4); m = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4); }
            const float d1 = d * (float) s, m1 = dmin * (float) m;
            const uint8_t* q = x + 16 + (j / 2) * 32;
            const bool high = j % 2 == 1;
            for (int t = 0; t < 4; ++t) {
                const int l = l0 + t;
                store(out, out_dt, v0 + t, d1 * (float) (high ? (q[l] >> 4) : (q[l] & 0xF)) - m1);
            }
        });
    } else if (qtype == 14) {   // Q6_K
        c.q.parallel_for(sycl::range<1>(items), [=](sycl::id<1> id) {
            const size_t v0 = id[0] * 4;
            const int64_t blk = (int64_t) (v0 / 256);
            const int j0 = (int) (v0 % 256), half = j0 / 128, k = (j0 % 128) / 32, l0 = j0 % 32;
            const uint8_t* x = b + blk * 210;
            const uint8_t* ql = x + half * 64;
            const uint8_t* qh = x + 128 + half * 32;
            const int8_t* sc = reinterpret_cast<const int8_t*>(x + 192) + half * 8;
            const float d = h2f(x + 208);
            for (int t = 0; t < 4; ++t) {
                const int l = l0 + t;
                const int lo = (k == 0) ? (ql[l] & 0xF) : (k == 1) ? (ql[l + 32] & 0xF) : (k == 2) ? (ql[l] >> 4) : (ql[l + 32] >> 4);
                const int hi = (qh[l] >> (2 * k)) & 3;
                const int q = (lo | (hi << 4)) - 32;
                store(out, out_dt, v0 + t, d * (float) sc[l / 16 + 2 * k] * (float) q);
            }
        });
    } else {
        g_err = "nsd_dequant: only Q4_K (12) and Q6_K (14)"; return -1;
    }
    return 0;
} catch (const std::exception& e) { g_err = e.what(); return -1; }

// Causal attention with grouped heads (Hq query heads share Hkv key/value heads), for short sequences: one
// work-group per (head, query), a lane per feature, an online softmax over the keys up to the query.
// q rows of stride qs, k and v rows of stride kvs (elements), out [L, Hq * D].
int nsd_attention_causal(void* ctx, const void* q, const void* k, const void* v, int dt, int64_t L, int64_t Hq, int64_t Hkv, int64_t D,
                         int64_t qs, int64_t kvs, void* out) try {
    auto& c = *static_cast<Ctx*>(ctx);
    if (L <= 0) return 0;
    if (D > 256 || Hq % Hkv != 0) { g_err = "nsd_attention_causal: D <= 256, Hq a multiple of Hkv"; return -1; }
    const int64_t wg = D <= 64 ? 64 : (D <= 128 ? 128 : 256);
    const float scale = 1.0f / sycl::sqrt((float) D);
    c.q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (Hq * L * wg)), sycl::range<1>((size_t) wg)), [=](sycl::nd_item<1> it) {
        const int64_t hi = it.get_group(0), h = hi / L, i = hi % L, d = it.get_local_id(0);
        const int64_t g = h / (Hq / Hkv);
        const float qd = d < D ? load(q, dt, (size_t) (i * qs + h * D + d)) : 0.0f;
        float m = -INFINITY, l = 0.0f, acc = 0.0f;
        for (int64_t j = 0; j <= i; ++j) {
            const float kd = d < D ? load(k, dt, (size_t) (j * kvs + g * D + d)) : 0.0f;
            const float s = sycl::reduce_over_group(it.get_group(), qd * kd, sycl::plus<float>()) * scale;
            const float mn = sycl::fmax(m, s);
            const float a = sycl::exp(m - mn), e = sycl::exp(s - mn);
            const float vd = d < D ? load(v, dt, (size_t) (j * kvs + g * D + d)) : 0.0f;
            acc = acc * a + e * vd;
            l = l * a + e;
            m = mn;
        }
        if (d < D) store(out, dt, (size_t) (i * Hq * D + h * D + d), acc / l);
    });
    return 0;
} catch (const std::exception& e) { g_err = e.what(); return -1; }

// ---- the latent upscaler: channels-last volumes [T, H, W, C], 16-bit

int nsd_conv3d(void* ctx, const void* x, int dt, int64_t T, int64_t H, int64_t W, int64_t Ci, const void* w, int64_t Co, int64_t k,
               const float* bias, void* out) try {
    auto& c = *static_cast<Ctx*>(ctx);
    if (T <= 0 || H <= 0 || W <= 0) return 0;
    using dnnl::memory;
    const int64_t p = k / 2;
    memory::desc smd({1, Ci, T, H, W}, ddt(dt), memory::format_tag::ndhwc);
    memory::desc dmd({1, Co, T, H, W}, ddt(dt), memory::format_tag::ndhwc);
    memory::desc wany({Co, Ci, k, k, k}, ddt(dt), memory::format_tag::any);
    memory::desc bmd({Co}, memory::data_type::f32, memory::format_tag::a);
    auto key = std::make_tuple(T, H, W, Ci, Co, k);
    auto it = c.conv3.find(key);
    if (it == c.conv3.end()) {
        auto pd = dnnl::convolution_forward::primitive_desc(c.eng, dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_direct,
                                                            smd, wany, bias ? bmd : memory::desc(), dmd, {1, 1, 1}, {p, p, p}, {p, p, p});
        it = c.conv3.emplace(key, Ctx::Conv3{dnnl::convolution_forward(pd), pd}).first;
    }
    auto wit = c.conv3_w.find(w);
    if (wit == c.conv3_w.end() || wit->second.get_desc() != it->second.pd.weights_desc()) {
        memory::desc wmd({Co, Ci, k, k, k}, ddt(dt), memory::format_tag::oidhw);
        memory wm(it->second.pd.weights_desc(), c.eng);
        memory src = usm(wmd, c.eng, w);
        dnnl::reorder(src, wm).execute(c.strm, src, wm);
        c.conv3_w[w] = wm;
        wit = c.conv3_w.find(w);
    }
    std::unordered_map<int, memory> args{{DNNL_ARG_SRC, usm(smd, c.eng, x)}, {DNNL_ARG_WEIGHTS, wit->second}, {DNNL_ARG_DST, usm(dmd, c.eng, out)}};
    if (bias) args.insert({DNNL_ARG_BIAS, usm(bmd, c.eng, bias)});
    it->second.prim.execute(c.strm, args);
    return 0;
} catch (const std::exception& e) { g_err = e.what(); return -1; }

// ---- the pixel upscaler (ESRGAN-type): frames channels-last [N, H, W, C], 16-bit
int nsd_conv2d(void* ctx, const void* x, int dt, int64_t N, int64_t H, int64_t W, int64_t Ci, const void* w, int64_t Co, int64_t k,
               const float* bias, void* out) try {
    auto& c = *static_cast<Ctx*>(ctx);
    if (N <= 0 || H <= 0 || W <= 0) return 0;
    using dnnl::memory;
    const int64_t p = k / 2;
    memory::desc smd({N, Ci, H, W}, ddt(dt), memory::format_tag::nhwc);
    memory::desc dmd({N, Co, H, W}, ddt(dt), memory::format_tag::nhwc);
    memory::desc wany({Co, Ci, k, k}, ddt(dt), memory::format_tag::any);
    memory::desc bmd({Co}, memory::data_type::f32, memory::format_tag::a);
    auto key = std::make_tuple(N, H, W, Ci, Co, k, dt);
    auto it = c.conv2.find(key);
    if (it == c.conv2.end()) {
        auto pd = dnnl::convolution_forward::primitive_desc(c.eng, dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_direct,
                                                            smd, wany, bias ? bmd : memory::desc(), dmd, {1, 1}, {p, p}, {p, p});
        it = c.conv2.emplace(key, Ctx::Conv3{dnnl::convolution_forward(pd), pd}).first;
    }
    auto wit = c.conv3_w.find(w);
    if (wit == c.conv3_w.end() || wit->second.get_desc() != it->second.pd.weights_desc()) {
        memory::desc wmd({Co, Ci, k, k}, ddt(dt), memory::format_tag::oihw);
        memory wm(it->second.pd.weights_desc(), c.eng);
        memory src = usm(wmd, c.eng, w);
        dnnl::reorder(src, wm).execute(c.strm, src, wm);
        c.conv3_w[w] = wm;
        wit = c.conv3_w.find(w);
    }
    std::unordered_map<int, memory> args{{DNNL_ARG_SRC, usm(smd, c.eng, x)}, {DNNL_ARG_WEIGHTS, wit->second}, {DNNL_ARG_DST, usm(dmd, c.eng, out)}};
    if (bias) args.insert({DNNL_ARG_BIAS, usm(bmd, c.eng, bias)});
    it->second.prim.execute(c.strm, args);
    return 0;
} catch (const std::exception& e) { g_err = e.what(); return -1; }

int nsd_prelu(void* ctx, void* x, int dt, int64_t M, int64_t C, const float* alpha) try {
    auto& c = *static_cast<Ctx*>(ctx);
    if (M <= 0 || C <= 0) return 0;
    c.q.parallel_for(sycl::range<2>((size_t) M, (size_t) C), [=](sycl::id<2> id) {
        const size_t i = id[0] * C + id[1];
        const float v = load(x, dt, i);
        store(x, dt, i, v > 0.0f ? v : alpha[id[1]] * v);
    });
    return 0;
} catch (const std::exception& e) { g_err = e.what(); return -1; }

int nsd_pixel_shuffle_add(void* ctx, const void* x, int dt, int64_t N, int64_t H, int64_t W, int64_t C, int64_t r,
                          const void* base, void* out) try {
    auto& c = *static_cast<Ctx*>(ctx);
    if (N <= 0 || H <= 0 || W <= 0) return 0;
    const int64_t Ho = H * r, Wo = W * r;
    c.q.parallel_for(sycl::range<3>((size_t) (N * Ho), (size_t) Wo, (size_t) C), [=](sycl::id<3> id) {
        const int64_t n = id[0] / Ho, yo = id[0] % Ho, xo = id[1], ch = id[2];
        const int64_t y = yo / r, i = yo % r, xx = xo / r, j = xo % r;
        const size_t src = ((size_t) ((n * H + y) * W + xx)) * (C * r * r) + (size_t) (ch * r * r + i * r + j);
        const size_t b = ((size_t) ((n * H + y) * W + xx)) * C + ch;
        store(out, dt, ((size_t) ((n * Ho + yo) * Wo + xo)) * C + ch, load(x, dt, src) + load(base, dt, b));
    });
    return 0;
} catch (const std::exception& e) { g_err = e.what(); return -1; }

int nsd_resize_area(void* ctx, const void* x, int dt, int64_t N, int64_t Hi, int64_t Wi, int64_t C, int64_t Ho, int64_t Wo,
                    float* out) try {
    auto& c = *static_cast<Ctx*>(ctx);
    if (N <= 0 || Ho <= 0 || Wo <= 0) return 0;
    c.q.parallel_for(sycl::range<3>((size_t) (C * N), (size_t) Ho, (size_t) Wo), [=](sycl::id<3> id) {
        const int64_t ch = id[0] / N, n = id[0] % N, yo = id[1], xo = id[2];
        // adaptive average pooling: [floor(o * In / Out), ceil((o + 1) * In / Out))
        const int64_t y0 = (yo * Hi) / Ho, y1 = ((yo + 1) * Hi + Ho - 1) / Ho;
        const int64_t x0 = (xo * Wi) / Wo, x1 = ((xo + 1) * Wi + Wo - 1) / Wo;
        float s = 0.0f;
        for (int64_t y = y0; y < y1; ++y)
            for (int64_t xx = x0; xx < x1; ++xx) s += load(x, dt, ((size_t) ((n * Hi + y) * Wi + xx)) * C + ch);
        const float v = s / (float) ((y1 - y0) * (x1 - x0));
        out[((size_t) (ch * N + n) * Ho + yo) * Wo + xo] = sycl::clamp(v, 0.0f, 1.0f);
    });
    return 0;
} catch (const std::exception& e) { g_err = e.what(); return -1; }

// GroupNorm over [N, C] (N voxels, channels last), G groups of C / G channels, statistics over every voxel of a
// group; then the affine weight and bias, optionally * (1 + scale[c]) + shift[c], then SiLU.
int nsd_group_norm_silu(void* ctx, const void* x, int dt, int64_t F, int64_t N, int64_t C, int64_t G, const float* weight, const float* bias,
                        float eps, const float* scale, const float* shift, void* out) try {
    auto& c = *static_cast<Ctx*>(ctx);
    if (F <= 0 || N <= 0 || C <= 0) return 0;
    const int64_t cg = C / G;
    constexpr int64_t kParts = 256;               // partial sums per (frame, group)
    float* gs = c.grow(c.gn, c.gn_cap, (size_t) (F * G * kParts * 2 + F * G * 2));
    if (!gs) return -1;
    float* stat = gs + F * G * kParts * 2;
    const int64_t per = (N + kParts - 1) / kParts;
    c.q.parallel_for(sycl::nd_range<2>(sycl::range<2>((size_t) (F * G * kParts), 64), sycl::range<2>(1, 64)), [=](sycl::nd_item<2> it) {
        const int64_t gp = it.get_global_id(0), fg = gp / kParts, part = gp % kParts, f = fg / G, g = fg % G;
        const int64_t lane = it.get_local_id(1);
        float s = 0.0f, s2 = 0.0f;
        const int64_t n0 = part * per, n1 = sycl::min(N, n0 + per);
        for (int64_t n = n0; n < n1; ++n)
            for (int64_t j = lane; j < cg; j += 64) {
                const float v = load(x, dt, (size_t) ((f * N + n) * C + g * cg + j));
                s += v; s2 += v * v;
            }
        s = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>());
        s2 = sycl::reduce_over_group(it.get_group(), s2, sycl::plus<float>());
        if (lane == 0) { gs[gp * 2] = s; gs[gp * 2 + 1] = s2; }
    });
    c.q.parallel_for(sycl::range<1>((size_t) (F * G)), [=](sycl::id<1> fg) {
        float s = 0.0f, s2 = 0.0f;
        for (int64_t p = 0; p < kParts; ++p) { s += gs[(fg * kParts + p) * 2]; s2 += gs[(fg * kParts + p) * 2 + 1]; }
        const float cnt = (float) N * (float) cg, mean = s / cnt, var = sycl::fmax(s2 / cnt - mean * mean, 0.0f);
        stat[fg * 2] = mean;
        stat[fg * 2 + 1] = sycl::rsqrt(var + eps);
    });
    c.q.parallel_for(sycl::range<2>((size_t) (F * N), (size_t) C), [=](sycl::id<2> id) {
        const size_t fn = id[0], ch = id[1], i = fn * C + ch;
        const int64_t g = (int64_t) (fn / N) * G + ch / cg;
        float v = (load(x, dt, i) - stat[g * 2]) * stat[g * 2 + 1] * weight[ch] + bias[ch];
        if (scale) v = v * (1.0f + scale[ch]) + shift[ch];
        store(out, dt, i, v / (1.0f + sycl::exp(-v)));
    });
    return 0;
} catch (const std::exception& e) { g_err = e.what(); return -1; }

// [T, H, W, C] -> [T + front, H + top + bottom, W + left + right, C]: zeros in front along T (causal), the spatial
// border reflected (PyTorch's "reflect": the edge row itself not repeated)
int nsd_pad3d(void* ctx, const void* x, int dt, int64_t T, int64_t H, int64_t W, int64_t C, int64_t front, int64_t top, int64_t bottom,
              int64_t left, int64_t right, void* out) try {
    auto& c = *static_cast<Ctx*>(ctx);
    const int64_t To = T + front, Ho = H + top + bottom, Wo = W + left + right;
    if (To <= 0 || Ho <= 0 || Wo <= 0 || C <= 0) return 0;
    c.q.parallel_for(sycl::range<2>((size_t) (To * Ho * Wo), (size_t) C), [=](sycl::id<2> id) {
        const int64_t o = id[0], ch = id[1];
        const int64_t t = o / (Ho * Wo) - front, h0 = (o / Wo) % Ho - top, w0 = o % Wo - left;
        float v = 0.0f;
        if (t >= 0) {
            const int64_t h = h0 < 0 ? -h0 : (h0 >= H ? 2 * (H - 1) - h0 : h0);
            const int64_t w = w0 < 0 ? -w0 : (w0 >= W ? 2 * (W - 1) - w0 : w0);
            v = load(x, dt, (size_t) (((t * H + h) * W + w) * C + ch));
        }
        store(out, dt, (size_t) (o * C + ch), v);
    });
    return 0;
} catch (const std::exception& e) { g_err = e.what(); return -1; }

// a kt x kh x kw convolution with strides and no padding (pad first: nsd_pad3d), channels last
int nsd_conv3d_ex(void* ctx, const void* x, int dt, int64_t T, int64_t H, int64_t W, int64_t Ci, const void* w, int64_t Co,
                  int64_t kt, int64_t kh, int64_t kw, int64_t st, int64_t sh, int64_t sw, const float* bias, void* out) try {
    auto& c = *static_cast<Ctx*>(ctx);
    const int64_t To = (T - kt) / st + 1, Ho = (H - kh) / sh + 1, Wo = (W - kw) / sw + 1;
    if (To <= 0 || Ho <= 0 || Wo <= 0) return 0;
    using dnnl::memory;
    memory::desc smd({1, Ci, T, H, W}, ddt(dt), memory::format_tag::ndhwc);
    memory::desc dmd({1, Co, To, Ho, Wo}, ddt(dt), memory::format_tag::ndhwc);
    memory::desc wany({Co, Ci, kt, kh, kw}, ddt(dt), memory::format_tag::any);
    memory::desc bmd({Co}, memory::data_type::f32, memory::format_tag::a);
    auto key = std::make_tuple(std::vector<int64_t>{T, H, W, Ci, Co, kt, kh, kw, st, sh, sw, bias ? 1 : 0, dt});
    auto it = c.conv3x.find(key);
    if (it == c.conv3x.end()) {
        auto pd = dnnl::convolution_forward::primitive_desc(c.eng, dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_direct,
                                                            smd, wany, bias ? bmd : memory::desc(), dmd, {st, sh, sw}, {0, 0, 0}, {0, 0, 0});
        it = c.conv3x.emplace(key, Ctx::Conv3{dnnl::convolution_forward(pd), pd}).first;
    }
    auto wit = c.conv3_w.find(w);
    if (wit == c.conv3_w.end() || wit->second.get_desc() != it->second.pd.weights_desc()) {
        memory::desc wmd({Co, Ci, kt, kh, kw}, ddt(dt), memory::format_tag::oidhw);
        memory wm(it->second.pd.weights_desc(), c.eng);
        memory src = usm(wmd, c.eng, w);
        dnnl::reorder(src, wm).execute(c.strm, src, wm);
        c.conv3_w[w] = wm;
        wit = c.conv3_w.find(w);
    }
    std::unordered_map<int, memory> args{{DNNL_ARG_SRC, usm(smd, c.eng, x)}, {DNNL_ARG_WEIGHTS, wit->second}, {DNNL_ARG_DST, usm(dmd, c.eng, out)}};
    if (bias) args.insert({DNNL_ARG_BIAS, usm(bmd, c.eng, bias)});
    it->second.prim.execute(c.strm, args);
    return 0;
} catch (const std::exception& e) { g_err = e.what(); return -1; }

// depthwise convolution along T only (kernel K, zero padding K / 2), x and out [T, P, C], w [C, K] float32
int nsd_temporal_dwconv(void* ctx, const void* x, int dt, int64_t T, int64_t P, int64_t C, const float* w, int64_t K,
                        const float* bias, void* out) try {
    auto& c = *static_cast<Ctx*>(ctx);
    if (T <= 0 || P <= 0 || C <= 0) return 0;
    c.q.parallel_for(sycl::range<3>((size_t) T, (size_t) P, (size_t) C), [=](sycl::id<3> id) {
        const int64_t t = id[0], p = id[1], ch = id[2];
        float acc = bias ? bias[ch] : 0.0f;
        for (int64_t k = 0; k < K; ++k) {
            const int64_t ts = t + k - K / 2;
            if (ts >= 0 && ts < T) acc += w[ch * K + k] * load(x, dt, (size_t) ((ts * P + p) * C + ch));
        }
        store(out, dt, (size_t) ((t * P + p) * C + ch), acc);
    });
    return 0;
} catch (const std::exception& e) { g_err = e.what(); return -1; }

// trilinear resize of [T, H, W, C] to [To, Ho, Wo, C] (PyTorch's align_corners=False)
int nsd_trilinear(void* ctx, const void* x, int dt, int64_t T, int64_t H, int64_t W, int64_t C, int64_t To, int64_t Ho, int64_t Wo,
                  void* out) try {
    auto& c = *static_cast<Ctx*>(ctx);
    if (To <= 0 || Ho <= 0 || Wo <= 0 || C <= 0) return 0;
    const float st = (float) T / (float) To, sh = (float) H / (float) Ho, sw = (float) W / (float) Wo;
    c.q.parallel_for(sycl::range<2>((size_t) (To * Ho * Wo), (size_t) C), [=](sycl::id<2> id) {
        const int64_t o = id[0], ch = id[1];
        const int64_t ot = o / (Ho * Wo), oh = (o / Wo) % Ho, ow = o % Wo;
        auto src = [](int64_t d, float scale, int64_t in, int64_t& i0, int64_t& i1, float& l) {
            float s = ((float) d + 0.5f) * scale - 0.5f;
            if (s < 0.0f) s = 0.0f;
            i0 = (int64_t) s;
            if (i0 > in - 1) i0 = in - 1;
            i1 = i0 < in - 1 ? i0 + 1 : i0;
            l = s - (float) i0;
        };
        int64_t t0, t1, h0, h1, w0, w1;
        float lt, lh, lw;
        src(ot, st, T, t0, t1, lt);
        src(oh, sh, H, h0, h1, lh);
        src(ow, sw, W, w0, w1, lw);
        auto v = [&](int64_t t, int64_t h, int64_t w) { return load(x, dt, (size_t) (((t * H + h) * W + w) * C + ch)); };
        const float a = (1 - lw) * v(t0, h0, w0) + lw * v(t0, h0, w1), b = (1 - lw) * v(t0, h1, w0) + lw * v(t0, h1, w1);
        const float cc = (1 - lw) * v(t1, h0, w0) + lw * v(t1, h0, w1), d = (1 - lw) * v(t1, h1, w0) + lw * v(t1, h1, w1);
        const float r = (1 - lt) * ((1 - lh) * a + lh * b) + lt * ((1 - lh) * cc + lh * d);
        store(out, dt, (size_t) (o * C + ch), r);
    });
    return 0;
} catch (const std::exception& e) { g_err = e.what(); return -1; }

// ---- the audio decoder (BigVGAN): float32 1-D convolutions over [B, C, L] signals

int nsd_conv1d(void* ctx, const float* x, int64_t B, int64_t Ci, int64_t L, const float* w, int64_t Co, int64_t K,
               const float* bias, int64_t stride, int64_t dil, int64_t pad, float* out, int64_t Lo) try {
    auto& c = *static_cast<Ctx*>(ctx);
    if (B <= 0 || Co <= 0 || Lo <= 0) return 0;
    c.q.parallel_for(sycl::range<3>((size_t) B, (size_t) Co, (size_t) Lo), [=](sycl::id<3> id) {
        const int64_t b = id[0], o = id[1], n = id[2];
        float acc = bias ? bias[o] : 0.0f;
        const int64_t base = n * stride - pad;
        for (int64_t i = 0; i < Ci; ++i) {
            const float* xr = x + (b * Ci + i) * L;
            const float* wr = w + (o * Ci + i) * K;
            for (int64_t k = 0; k < K; ++k) {
                const int64_t p = base + k * dil;
                if (p >= 0 && p < L) acc += wr[k] * xr[p];
            }
        }
        out[(b * Co + o) * Lo + n] = acc;
    });
    return 0;
} catch (const std::exception& e) { g_err = e.what(); return -1; }

int nsd_conv_transpose1d(void* ctx, const float* x, int64_t B, int64_t Ci, int64_t L, const float* w, int64_t Co, int64_t K,
                         const float* bias, int64_t stride, int64_t pad, float* out, int64_t Lo) try {
    auto& c = *static_cast<Ctx*>(ctx);
    if (B <= 0 || Co <= 0 || Lo <= 0) return 0;
    c.q.parallel_for(sycl::range<3>((size_t) B, (size_t) Co, (size_t) Lo), [=](sycl::id<3> id) {
        const int64_t b = id[0], o = id[1], m = id[2];
        float acc = bias ? bias[o] : 0.0f;
        // out[m] gathers x[n] w[k] over m = n * stride - pad + k
        const int64_t mp = m + pad;
        for (int64_t k = mp % stride; k < K; k += stride) {
            const int64_t n = (mp - k) / stride;
            if (n < 0 || n >= L) continue;
            for (int64_t i = 0; i < Ci; ++i) acc += x[(b * Ci + i) * L + n] * w[(i * Co + o) * K + k];
        }
        out[(b * Co + o) * Lo + m] = acc;
    });
    return 0;
} catch (const std::exception& e) { g_err = e.what(); return -1; }

// BigVGAN's anti-aliased SnakeBeta in one pass: upsample x2 (12-tap transposed filter, replicate padding), x +
// sin^2(alpha x) / beta, low-pass and downsample x2 (12-tap filter, replicate padding). Per output sample: 12
// upsampled values, each from 6 input taps.
int nsd_aa_snake(void* ctx, const float* x, int64_t B, int64_t C, int64_t L, const float* log_alpha, const float* log_beta,
                 const float* up, const float* down, float* out) try {
    auto& c = *static_cast<Ctx*>(ctx);
    if (B <= 0 || C <= 0 || L <= 0) return 0;
    c.q.parallel_for(sycl::range<3>((size_t) B, (size_t) C, (size_t) L), [=](sycl::id<3> id) {
        const int64_t b = id[0], ch = id[1], n = id[2];
        const float* xr = x + (b * C + ch) * L;
        const float alpha = sycl::exp(log_alpha[ch]), inv_beta = 1.0f / (sycl::exp(log_beta[ch]) + 1e-9f);
        float acc = 0.0f;
        for (int k = 0; k < 12; ++k) {
            // the upsampled signal (length 2L) at j, replicate-padded by 5 on the left
            int64_t j = 2 * n + k - 5;
            j = j < 0 ? 0 : (j > 2 * L - 1 ? 2 * L - 1 : j);
            float u = 0.0f;
            // up[j] = 2 sum_k' f[k'] xp[(j + 15 - k') / 2] over even j + 15 - k'; xp[p] = x[clamp(p - 5)]
            for (int kk = (int) ((j + 15) & 1); kk < 12; kk += 2) {
                int64_t p = (j + 15 - kk) / 2 - 5;
                p = p < 0 ? 0 : (p > L - 1 ? L - 1 : p);
                u += up[kk] * xr[p];
            }
            u *= 2.0f;
            const float s = sycl::sin(alpha * u);
            acc += down[k] * (u + inv_beta * s * s);
        }
        out[(b * C + ch) * L + n] = acc;
    });
    return 0;
} catch (const std::exception& e) { g_err = e.what(); return -1; }

// the DAC encoder's Snake: x + sin^2(alpha x) / alpha, alpha per channel (as stored, not in log scale)
int nsd_snake(void* ctx, const float* x, int64_t B, int64_t C, int64_t L, const float* alpha, float* out) try {
    auto& c = *static_cast<Ctx*>(ctx);
    if (B <= 0 || C <= 0 || L <= 0) return 0;
    c.q.parallel_for(sycl::range<1>((size_t) (B * C * L)), [=](sycl::id<1> i) {
        const float a = alpha[(i[0] / L) % C], v = x[i[0]], s = sycl::sin(a * v);
        out[i[0]] = v + s * s / (a + 1e-9f);
    });
    return 0;
} catch (const std::exception& e) { g_err = e.what(); return -1; }

int nsd_scale(void* ctx, float* x, int64_t n, float s) try {
    auto& c = *static_cast<Ctx*>(ctx);
    if (n <= 0) return 0;
    c.q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) { x[i] *= s; });
    return 0;
} catch (const std::exception& e) { g_err = e.what(); return -1; }

int nsd_layer_norm(void* ctx, const void* x, int x_dt, int64_t M, int64_t C, const float* weight, const float* bias, float eps,
                   void* out, int out_dt) try {
    auto& c = *static_cast<Ctx*>(ctx);
    if (M <= 0 || C <= 0) return 0;
    float* st = c.grow(c.inv, c.inv_cap, (size_t) M * 2);   // (mean, 1 / std) per row
    if (!st) return -1;
    c.q.parallel_for(sycl::range<1>((size_t) M), [=](sycl::id<1> r) {
        float s = 0.0f;
        for (int64_t i = 0; i < C; ++i) s += load(x, x_dt, r[0] * C + i);
        const float mean = s / (float) C;
        float v2 = 0.0f;
        for (int64_t i = 0; i < C; ++i) {
            const float d = load(x, x_dt, r[0] * C + i) - mean;
            v2 += d * d;
        }
        st[2 * r[0]] = mean;
        st[2 * r[0] + 1] = sycl::rsqrt(v2 / (float) C + eps);
    });
    c.q.parallel_for(sycl::range<2>((size_t) M, (size_t) C), [=](sycl::id<2> id) {
        const size_t r = id[0], i = id[1];
        float v = (load(x, x_dt, r * C + i) - st[2 * r]) * st[2 * r + 1];
        if (weight) v *= weight[i];
        if (bias) v += bias[i];
        store(out, out_dt, r * C + i, v);
    });
    return 0;
} catch (const std::exception& e) { g_err = e.what(); return -1; }

int nsd_rms_rope(void* ctx, void* x, int x_dt, int64_t M, int64_t H, int64_t D, int64_t stride, const float* weight,
                 float eps, const float* cs, int rot_dim) try {
    auto& c = *static_cast<Ctx*>(ctx);
    if (M <= 0 || H <= 0 || D <= 0) return 0;
    if (rot_dim < 0 || rot_dim > D || rot_dim % 2 != 0 || (D - rot_dim) % 2 != 0) {
        g_err = "nsd_rms_rope: rot_dim must be even, at most D, and leave an even remainder"; return -1;
    }
    if (stride < H * D) { g_err = "nsd_rms_rope: the row stride is shorter than a row"; return -1; }
    float* inv = c.grow(c.inv, c.inv_cap, (size_t) (M * H));
    if (!inv) return -1;
    // 16-bit rows 16-byte aligned: each head's features read 8 at a time, summed in the same order (identical results)
    const bool vec = (x_dt == F16 || x_dt == BF16) && D % 8 == 0 && stride % 8 == 0 && (uintptr_t) x % 16 == 0 &&
                     !std::getenv("NSD_NORM_SCALAR");
    if (vec) {
        const bool f16 = x_dt == F16;
        c.q.parallel_for(sycl::range<1>((size_t) (M * H)), [=](sycl::id<1> mh) {
            const size_t base = (mh[0] / H) * stride + (mh[0] % H) * D;
            using V = sycl::vec<uint16_t, 8>;
            float s = 0.0f;
            for (int64_t i = 0; i < D; i += 8) {
                const V xv = *reinterpret_cast<const V*>((const uint16_t*) x + base + i);
                for (int e = 0; e < 8; ++e) {
                    const float v = f16 ? (float) sycl::bit_cast<sycl::half>((uint16_t) xv[e]) : bf16_to_f32(xv[e]);
                    s += v * v;
                }
            }
            inv[mh[0]] = sycl::rsqrt(s / (float) D + eps);
        });
    } else {
        c.q.parallel_for(sycl::range<1>((size_t) (M * H)), [=](sycl::id<1> mh) {
            const size_t base = (mh[0] / H) * stride + (mh[0] % H) * D;
            float s = 0.0f;
            for (int64_t i = 0; i < D; ++i) {
                const float v = load(x, x_dt, base + i);
                s += v * v;
            }
            inv[mh[0]] = sycl::rsqrt(s / (float) D + eps);
        });
    }
    const int64_t half = rot_dim / 2;                  // rotated pairs: (i, half + i)
    const int64_t slots = half + (D - rot_dim) / 2;    // then pairs of the features passed through
    // 16-bit, aligned: a work-item takes 8 rotated pairs or 16 features passed through, through 16-byte loads; the
    // same arithmetic per element (identical results)
    if (vec && half % 8 == 0 && (D - rot_dim) % 16 == 0) {
        const bool f16 = x_dt == F16;
        const int64_t g_rot = half / 8, g_pass = (D - rot_dim) / 16;
        c.q.parallel_for(sycl::range<2>((size_t) (M * H), (size_t) (g_rot + g_pass)), [=](sycl::id<2> id) {
            using V = sycl::vec<uint16_t, 8>;
            const size_t mh = id[0], g = id[1], base = (mh / H) * stride + (mh % H) * D;
            const float s = inv[mh];
            uint16_t* p = (uint16_t*) x;
            auto f = [=](uint16_t b) { return f16 ? (float) sycl::bit_cast<sycl::half>(b) : bf16_to_f32(b); };
            auto h = [=](float v) { return f16 ? sycl::bit_cast<uint16_t>((sycl::half) v) : f32_to_bf16(v); };
            if ((int64_t) g < g_rot) {
                const size_t j0 = g * 8;
                V* pa = reinterpret_cast<V*>(p + base + j0);
                V* pb = reinterpret_cast<V*>(p + base + half + j0);
                const V va = *pa, vb = *pb;
                V oa, ob;
                for (int e = 0; e < 8; ++e) {
                    const size_t j = j0 + e;
                    const float a = f(va[e]) * s * weight[j], b = f(vb[e]) * s * weight[half + j];
                    const size_t tt = (mh / H) * half + j;
                    const float co = cs[2 * tt], si = cs[2 * tt + 1];
                    oa[e] = h(a * co - b * si);
                    ob[e] = h(b * co + a * si);
                }
                *pa = oa;
                *pb = ob;
            } else {
                const size_t i0 = rot_dim + (g - g_rot) * 16;
                for (int k = 0; k < 2; ++k) {
                    V* pv = reinterpret_cast<V*>(p + base + i0 + 8 * k);
                    const V v = *pv;
                    V o;
                    for (int e = 0; e < 8; ++e) o[e] = h(f(v[e]) * s * weight[i0 + 8 * k + e]);
                    *pv = o;
                }
            }
        });
        return 0;
    }
    // one work-item per pair: it owns its two values, so the update is in place
    c.q.parallel_for(sycl::range<2>((size_t) (M * H), (size_t) slots), [=](sycl::id<2> id) {
        const size_t mh = id[0], j = id[1], base = (mh / H) * stride + (mh % H) * D;
        const float s = inv[mh];
        if ((int64_t) j < half) {
            const size_t ia = base + j, ib = base + half + j;
            const float a = load(x, x_dt, ia) * s * weight[j], b = load(x, x_dt, ib) * s * weight[half + j];
            const size_t t = (mh / H) * half + j;      // the token's angle for this pair
            const float co = cs[2 * t], si = cs[2 * t + 1];
            store(x, x_dt, ia, a * co - b * si);
            store(x, x_dt, ib, b * co + a * si);
        } else {
            const size_t i0 = rot_dim + 2 * (j - half);
            store(x, x_dt, base + i0, load(x, x_dt, base + i0) * s * weight[i0]);
            store(x, x_dt, base + i0 + 1, load(x, x_dt, base + i0 + 1) * s * weight[i0 + 1]);
        }
    });
    return 0;
} catch (const std::exception& e) { g_err = e.what(); return -1; }

// 16-bit in and out, 8 features per work-item through 16-byte loads and stores: the same arithmetic and rounding per
// element as the general kernel below (bit-identical results), at about the memory's rate - the general one, an
// element per work-item through a type switch, ran at a third of it (10% of the video decoder's time).
extern "C++" {
template <int DT>
static void swiglu16(sycl::queue& q, const uint16_t* x, int64_t M, int64_t C, uint16_t* out) {
    const size_t C8 = (size_t) C / 8;
    q.parallel_for(sycl::range<2>((size_t) M, C8), [=](sycl::id<2> id) {
        const size_t r = id[0], j = id[1] * 8;
        using V = sycl::vec<uint16_t, 8>;
        const V gv = *reinterpret_cast<const V*>(x + r * 2 * C + j);
        const V uv = *reinterpret_cast<const V*>(x + r * 2 * C + C + j);
        V ov;
        for (int e = 0; e < 8; ++e) {
            const uint16_t gb = gv[e], ub = uv[e];
            const float g = DT == F16 ? (float) sycl::bit_cast<sycl::half>(gb) : bf16_to_f32(gb);
            const float u = DT == F16 ? (float) sycl::bit_cast<sycl::half>(ub) : bf16_to_f32(ub);
            const float o = g / (1.0f + sycl::exp(-g)) * u;
            ov[e] = DT == F16 ? sycl::bit_cast<uint16_t>((sycl::half) o) : f32_to_bf16(o);
        }
        *reinterpret_cast<V*>(out + r * C + j) = ov;
    });
}
}  // extern "C++"

int nsd_swiglu(void* ctx, const void* x, int x_dt, int64_t M, int64_t C, void* out, int out_dt) try {
    auto& c = *static_cast<Ctx*>(ctx);
    if (M <= 0 || C <= 0) return 0;
    // the fast form needs 16-byte aligned rows: C a multiple of 8, the buffers 16-byte aligned (USM allocations are)
    const bool aligned = ((uintptr_t) x % 16 == 0) && ((uintptr_t) out % 16 == 0) && C % 8 == 0;
    if (aligned && x_dt == out_dt && (x_dt == F16 || x_dt == BF16) && !std::getenv("NSD_SWIGLU_SCALAR")) {
        if (x_dt == F16) swiglu16<F16>(c.q, (const uint16_t*) x, M, C, (uint16_t*) out);
        else swiglu16<BF16>(c.q, (const uint16_t*) x, M, C, (uint16_t*) out);
        return 0;
    }
    c.q.parallel_for(sycl::range<2>((size_t) M, (size_t) C), [=](sycl::id<2> id) {
        const size_t r = id[0], i = id[1];
        const float g = load(x, x_dt, r * 2 * C + i), u = load(x, x_dt, r * 2 * C + C + i);
        store(out, out_dt, r * C + i, g / (1.0f + sycl::exp(-g)) * u);
    });
    return 0;
} catch (const std::exception& e) { g_err = e.what(); return -1; }

int nsd_gate_add(void* ctx, void* x, int x_dt, int64_t M, int64_t C, const void* other, int other_dt,
                 const int32_t* rows, const float* gate) try {
    auto& c = *static_cast<Ctx*>(ctx);
    if (M <= 0 || C <= 0) return 0;
    c.q.parallel_for(sycl::range<2>((size_t) M, (size_t) C), [=](sycl::id<2> id) {
        const size_t r = id[0], i = id[1], p = r * C + i;
        const float g = gate ? gate[(size_t) rows[r] * C + i] : 1.0f;
        store(x, x_dt, p, load(x, x_dt, p) + load(other, other_dt, p) * g);
    });
    return 0;
} catch (const std::exception& e) { g_err = e.what(); return -1; }

// Attention, a chunk of query rows at a time. Two forms; both keep what is in device memory bounded whatever S is.
//
// 1. attention_fused - oneDNN's fused kernel: scores, softmax and the weighted sum over tiles that stay in the GPU's
//    registers, so no score table is written at all. About 5x the speed of form 2. oneDNN reaches it only through
//    its graph interface, and only for one exact form of the pattern (found with kernels/sdpa_probe.cpp and oneDNN's
//    sdp_primitive_config.cpp):
//        MatMul(q, k, transpose_b) -> Multiply(scale) -> SoftMax(axis -1, mode inf_as_zero) -> MatMul(., v)
//    with q, k, v, the weights and the result 16-bit, the two tensors in between declared FLOAT32, and the scale a
//    float32 HOST scalar; and it needs oneDNN 3.12 (3.11 has no such kernel on this card).
//    When oneDNN does not take that form as its fused kernel it runs the same graph as separate steps with the
//    whole score table of the call in device memory, and nothing tells the caller which happened. Called with all
//    S query rows at 16.5k tokens that table is 30 GiB; the xe driver has no out-of-memory error, the card spilled
//    into host RAM and the machine went down (2026-10-02). Two defences:
//      - built against the image's oneDNN (patched: container/onednn-sdpa-no-fallback.patch, -DNSD_SDPA_NO_FALLBACK),
//        oneDNN is told to fail instead of falling back, and form 2 takes over;
//      - built against any other oneDNN, the rows per call are chosen so that even the fallback's tables (10 bytes
//        per score) stay under ~1.5 GiB - safe, but the small chunks cost about a third of the speed.
//
// 2. attention_split - ours, used when oneDNN refuses form 1 (an older oneDNN):
//      scores   q . k^T               oneDNN, on the matrix engine          -> half [H, rows, S]
//      weights  exp(score - row max)  ours, in place, one score per work-item (the row's max from a read-only pass)
//      values   weights . [v | 1]     oneDNN, on the matrix engine          -> float32 [H, rows, D + 1]
//      out      values / row sum      ours, folded into the copy back to token order
//    (The column of ones beside v makes the matrix engine deliver each row's sum of weights with the values.
//    oneDNN's stand-alone softmax is not used: it took 1.0 s of a 1.2 s call on rows 16.5k long.)
//
// q, k, v are stored by token; both forms want them by head, so k and v are copied once per call and q a chunk at a
// time, in IEEE half (11 significant bits where bfloat16 has 8).
//
// NSD_PROFILE=1 in the environment: wait after every phase and report where a call's time went (slower).

static Ctx::Sdpa& sdpa_for(Ctx& c, int64_t rows, int64_t S, int64_t H, int64_t D) {
    auto key = std::make_tuple(rows, S, H, D);
    auto it = c.sdpa.find(key);
    if (it != c.sdpa.end()) return it->second;
    using namespace dnnl::graph;
    using lt = logical_tensor;
    const auto f16 = lt::data_type::f16, f32 = lt::data_type::f32;
    const auto strided = lt::layout_type::strided;
    size_t id = 0;
    lt q(id++, f16, lt::dims {1, H, rows, D}, strided), k(id++, f16, lt::dims {1, H, S, D}, strided);
    lt scale(id++, f32, lt::dims {}, strided, lt::property_type::host_scalar);
    lt v(id++, f16, lt::dims {1, H, S, D}, strided);
    lt score(id++, f32, lt::dims {1, H, rows, S}, strided), scaled(id++, f32, lt::dims {1, H, rows, S}, strided);
    lt probs(id++, f16, lt::dims {1, H, rows, S}, strided), out(id++, f16, lt::dims {1, H, rows, D}, strided);
    op bmm1(id++, op::kind::MatMul, {q, k}, {score}, "scores");
    bmm1.set_attr<bool>(op::attr::transpose_b, true);
    op mul(id++, op::kind::Multiply, {score, scale}, {scaled}, "scale");
    op sm(id++, op::kind::SoftMax, {scaled}, {probs}, "softmax");
    sm.set_attr<int64_t>(op::attr::axis, -1);
    sm.set_attr<std::string>(op::attr::mode, "inf_as_zero");
    op bmm2(id++, op::kind::MatMul, {probs, v}, {out}, "values");
    graph g(dnnl::engine::kind::gpu);
    g.add_op(bmm1); g.add_op(mul); g.add_op(sm); g.add_op(bmm2);
    g.finalize();
    auto parts = g.get_partitions();
    if (parts.size() != 1 || !parts[0].is_supported())
        throw std::runtime_error("oneDNN did not take attention as one partition");
    const lt mine[4] = {q, k, scale, v};
    Ctx::Sdpa sd;
    for (const auto& port : parts[0].get_input_ports())
        for (int i = 0; i < 4; ++i)
            if (mine[i].get_id() == port.get_id()) { sd.in.push_back(mine[i]); sd.slot.push_back(i); }
    if (sd.in.size() != 4) throw std::runtime_error("oneDNN's attention partition has unexpected inputs");
    sd.cp = parts[0].compile(sd.in, {out}, c.eng);
    sd.out = sd.cp.query_logical_tensor(out.get_id());
    return c.sdpa.emplace(key, std::move(sd)).first->second;
}

// Form 1. Returns false (having queued nothing) when oneDNN refuses the pattern; throws on anything else.
static bool attention_fused(Ctx& c, const void* q, const void* k, const void* v, int dt, int64_t S, int64_t H, int64_t D,
                            int64_t stride, void* out, int out_dt) {
    sycl::queue& qu = c.q;
#ifdef NSD_SDPA_NO_FALLBACK
    // This oneDNN cannot fall back (see init): the fused kernel or an error. So the chunk is sized for speed alone -
    // measured at 16.5k keys: 174 rows per call 156 ms, 464 rows 99 ms, 1859 rows 89 ms (and see attn_rows).
    const int64_t rows_max = std::min<int64_t>(S, c.attn_rows);
#else
    // rows per chunk: if oneDNN falls back to separate steps it holds scores and scaled scores in float32 and the
    // weights in half - 10 bytes per score. Keep that under the table budget (1.5 GiB unless NSD_ATTN_TABLE_MB says
    // otherwise).
    const int64_t rows_max = std::max<int64_t>(1, std::min<int64_t>(S, (int64_t) c.attn_table_bytes / (H * S * 10)));
#endif
    try {
        sdpa_for(c, std::min(rows_max, S), S, H, D);
        if (S % rows_max != 0 && S > rows_max) sdpa_for(c, S % rows_max, S, H, D);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "h3sycl: oneDNN's fused attention is not available (%s); using the split form\n", e.what());
        return false;
    }
    const size_t n = (size_t) S * H * D;
    uint16_t* hq = c.grow(c.hq, c.hq_cap, (size_t) H * rows_max * D);
    uint16_t* hk = c.grow(c.hk, c.hk_cap, n);
    uint16_t* hv = c.grow(c.hv, c.hv_cap, n);
    float* aof = c.grow(c.ao, c.ao_cap, (size_t) H * rows_max * D);     // used as half [H, rows, D] here
    if (!hq || !hk || !hv || !aof) throw std::runtime_error(g_err);
    sycl::half* ao = (sycl::half*) aof;
    c.sdpa_scale = 1.0f / std::sqrt((float) D);

    const bool prof = c.profile;
    double t_ph[3] = {0, 0, 0};
    auto clock = std::chrono::steady_clock::now();
    auto lap = [&](int i) {
        if (!prof) return;
        qu.wait();
        const auto now = std::chrono::steady_clock::now();
        t_ph[i] += std::chrono::duration<double, std::milli>(now - clock).count();
        clock = now;
    };
    lap(0);
    qu.parallel_for(sycl::range<3>((size_t) H, (size_t) S, (size_t) D), [=](sycl::id<3> id) {
        const size_t src = id[1] * stride + id[0] * D + id[2], dst = (id[0] * S + id[1]) * D + id[2];
        ((sycl::half*) hk)[dst] = (sycl::half) load(k, dt, src);
        ((sycl::half*) hv)[dst] = (sycl::half) load(v, dt, src);
    });
    for (int64_t r0 = 0; r0 < S; r0 += rows_max) {
        const int64_t rows = std::min(rows_max, S - r0);
        qu.parallel_for(sycl::range<3>((size_t) H, (size_t) rows, (size_t) D), [=](sycl::id<3> id) {
            ((sycl::half*) hq)[(id[0] * rows + id[1]) * D + id[2]] = (sycl::half) load(q, dt, (r0 + id[1]) * stride + id[0] * D + id[2]);
        });
        Ctx::Sdpa& sd = sdpa_for(c, rows, S, H, D);
        void* const handles[4] = {hq, hk, nullptr, hv};
        std::vector<dnnl::graph::tensor> in;
        for (size_t i = 0; i < sd.in.size(); ++i) {
            if (sd.slot[i] == 2) in.push_back(dnnl::graph::tensor::make_scalar_tensor(sd.in[i], &c.sdpa_scale));
            else in.emplace_back(sd.in[i], c.eng, handles[sd.slot[i]]);
        }
        lap(0);
        sd.cp.execute(c.strm, in, {dnnl::graph::tensor(sd.out, c.eng, ao)});
        lap(1);
        // back to token order: out [S, H * D]
        qu.parallel_for(sycl::range<3>((size_t) rows, (size_t) H, (size_t) D), [=](sycl::id<3> id) {
            store(out, out_dt, ((r0 + id[0]) * H + id[1]) * D + id[2], (float) ao[(id[1] * rows + id[0]) * D + id[2]]);
        });
        lap(2);
    }
    if (prof)
        std::fprintf(stderr, "h3sycl: attention (fused) S=%lld, %lld rows per chunk: copies %.1f ms, attention %.1f, out %.1f\n",
                     (long long) S, (long long) rows_max, t_ph[0], t_ph[1], t_ph[2]);
    return true;
}

static int attention_split(Ctx& c, const void* q, const void* k, const void* v, int dt, int64_t S, int64_t H, int64_t D,
                           int64_t stride, void* out, int out_dt);

// nextsycl: Sq query rows over Skv key / value rows (a cached prefix and the rows' own block), the same fused graph
// a chunk of query rows at a time; q rows qs apart, k / v rows kvs apart, out [Sq, H * D]. No split fallback: an error
// when oneDNN refuses the pattern.
// nextsycl: the fused graph over q, k, v in their token-order buffers (rows qs / kvs apart, heads D apart) and out
// [rows, H * D] in token order - no copies by head
static Ctx::Sdpa& sdpa_qk_for(Ctx& c, int64_t rows, int64_t Skv, int64_t H, int64_t D, int64_t qs, int64_t kvs) {
    auto key = std::make_tuple(rows, Skv, H, D, qs, kvs);
    auto it = c.sdpa_qk.find(key);
    if (it != c.sdpa_qk.end()) return it->second;
    using namespace dnnl::graph;
    using lt = logical_tensor;
    const auto f16 = lt::data_type::f16, f32 = lt::data_type::f32;
    const auto strided = lt::layout_type::strided;
    size_t id = 0;
    lt q(id++, f16, lt::dims {1, H, rows, D}, lt::dims {rows * qs, D, qs, 1});
    lt k(id++, f16, lt::dims {1, H, Skv, D}, lt::dims {Skv * kvs, D, kvs, 1});
    lt scale(id++, f32, lt::dims {}, strided, lt::property_type::host_scalar);
    lt v(id++, f16, lt::dims {1, H, Skv, D}, lt::dims {Skv * kvs, D, kvs, 1});
    lt score(id++, f32, lt::dims {1, H, rows, Skv}, strided), scaled(id++, f32, lt::dims {1, H, rows, Skv}, strided);
    lt probs(id++, f16, lt::dims {1, H, rows, Skv}, strided);
    lt out(id++, f16, lt::dims {1, H, rows, D}, lt::dims {rows * H * D, D, H * D, 1});
    op bmm1(id++, op::kind::MatMul, {q, k}, {score}, "scores");
    bmm1.set_attr<bool>(op::attr::transpose_b, true);
    op mul(id++, op::kind::Multiply, {score, scale}, {scaled}, "scale");
    op sm(id++, op::kind::SoftMax, {scaled}, {probs}, "softmax");
    sm.set_attr<int64_t>(op::attr::axis, -1);
    sm.set_attr<std::string>(op::attr::mode, "inf_as_zero");
    op bmm2(id++, op::kind::MatMul, {probs, v}, {out}, "values");
    graph g(dnnl::engine::kind::gpu);
    g.add_op(bmm1); g.add_op(mul); g.add_op(sm); g.add_op(bmm2);
    g.finalize();
    auto parts = g.get_partitions();
    if (parts.size() != 1 || !parts[0].is_supported())
        throw std::runtime_error("oneDNN did not take the strided attention as one partition");
    const lt mine[4] = {q, k, scale, v};
    Ctx::Sdpa sd;
    for (const auto& port : parts[0].get_input_ports())
        for (int i = 0; i < 4; ++i)
            if (mine[i].get_id() == port.get_id()) { sd.in.push_back(mine[i]); sd.slot.push_back(i); }
    if (sd.in.size() != 4) throw std::runtime_error("oneDNN's strided attention partition has unexpected inputs");
    sd.cp = parts[0].compile(sd.in, {out}, c.eng);
    sd.out = sd.cp.query_logical_tensor(out.get_id());
    return c.sdpa_qk.emplace(key, std::move(sd)).first->second;
}

static void attention_fused_qk(Ctx& c, const void* q, int64_t Sq, int64_t qs, const void* k, const void* v, int64_t Skv,
                               int64_t kvs, int dt, int64_t H, int64_t D, void* out, int out_dt) {
    sycl::queue& qu = c.q;
    c.sdpa_scale = 1.0f / std::sqrt((float) D);
    if (dt == NSD_F16 && out_dt == NSD_F16 && c.sdpa_qk_direct_ok) {
        try {
            const int64_t rows_max = std::min<int64_t>(Sq, c.attn_rows);
            for (int64_t r0 = 0; r0 < Sq; r0 += rows_max) {
                const int64_t rows = std::min(rows_max, Sq - r0);
                Ctx::Sdpa& sd = sdpa_qk_for(c, rows, Skv, H, D, qs, kvs);
                void* const handles[4] = {(char*) q + (size_t) r0 * qs * 2, (void*) k, nullptr, (void*) v};
                std::vector<dnnl::graph::tensor> in;
                for (size_t i = 0; i < sd.in.size(); ++i) {
                    if (sd.slot[i] == 2) in.push_back(dnnl::graph::tensor::make_scalar_tensor(sd.in[i], &c.sdpa_scale));
                    else in.emplace_back(sd.in[i], c.eng, handles[sd.slot[i]]);
                }
                sd.cp.execute(c.strm, in, {dnnl::graph::tensor(sd.out, c.eng, (char*) out + (size_t) r0 * H * D * 2)});
            }
            return;
        } catch (const std::exception& e) {
            // nothing was queued for a chunk that failed to compile; the chunks before it are redone below
            c.sdpa_qk_direct_ok = false;
            std::fprintf(stderr, "nsd: the strided attention is not available (%s); copying by head\n", e.what());
        }
    }
    const int64_t rows_max = std::min<int64_t>(Sq, c.attn_rows);
    sdpa_for(c, rows_max, Skv, H, D);
    const size_t n = (size_t) Skv * H * D;
    uint16_t* hq = c.grow(c.hq, c.hq_cap, (size_t) H * rows_max * D);
    uint16_t* hk = c.grow(c.hk, c.hk_cap, n);
    uint16_t* hv = c.grow(c.hv, c.hv_cap, n);
    float* aof = c.grow(c.ao, c.ao_cap, (size_t) H * rows_max * D);
    if (!hq || !hk || !hv || !aof) throw std::runtime_error(g_err);
    sycl::half* ao = (sycl::half*) aof;
    c.sdpa_scale = 1.0f / std::sqrt((float) D);
    qu.parallel_for(sycl::range<3>((size_t) H, (size_t) Skv, (size_t) D), [=](sycl::id<3> id) {
        const size_t src = id[1] * kvs + id[0] * D + id[2], dst = (id[0] * Skv + id[1]) * D + id[2];
        ((sycl::half*) hk)[dst] = (sycl::half) load(k, dt, src);
        ((sycl::half*) hv)[dst] = (sycl::half) load(v, dt, src);
    });
    for (int64_t r0 = 0; r0 < Sq; r0 += rows_max) {
        const int64_t rows = std::min(rows_max, Sq - r0);
        qu.parallel_for(sycl::range<3>((size_t) H, (size_t) rows, (size_t) D), [=](sycl::id<3> id) {
            ((sycl::half*) hq)[(id[0] * rows + id[1]) * D + id[2]] = (sycl::half) load(q, dt, (r0 + id[1]) * qs + id[0] * D + id[2]);
        });
        Ctx::Sdpa& sd = sdpa_for(c, rows, Skv, H, D);
        void* const handles[4] = {hq, hk, nullptr, hv};
        std::vector<dnnl::graph::tensor> in;
        for (size_t i = 0; i < sd.in.size(); ++i) {
            if (sd.slot[i] == 2) in.push_back(dnnl::graph::tensor::make_scalar_tensor(sd.in[i], &c.sdpa_scale));
            else in.emplace_back(sd.in[i], c.eng, handles[sd.slot[i]]);
        }
        sd.cp.execute(c.strm, in, {dnnl::graph::tensor(sd.out, c.eng, ao)});
        qu.parallel_for(sycl::range<3>((size_t) rows, (size_t) H, (size_t) D), [=](sycl::id<3> id) {
            store(out, out_dt, ((r0 + id[0]) * H + id[1]) * D + id[2], (float) ao[(id[1] * rows + id[0]) * D + id[2]]);
        });
    }
}

// The fused graph for B sequences at once: q, k, v [B, H, S, D] and out [B, H, S, D] f16. In place (`direct`): the
// tensors are the callers' own token-order buffers seen through strides - row s of sequence b, head h at
// b * S * stride + s * stride + h * D (q, k, v) and (b * S + s) * H * D + h * D (out). Otherwise dense by head.
static Ctx::Sdpa& sdpa_batch_for(Ctx& c, int64_t B, int64_t S, int64_t H, int64_t D, int64_t stride, bool direct) {
    auto key = std::make_tuple(B, S, H, D, direct ? stride : 0, direct);
    auto it = c.sdpa_b.find(key);
    if (it != c.sdpa_b.end()) return it->second;
    using namespace dnnl::graph;
    using lt = logical_tensor;
    const auto f16 = lt::data_type::f16, f32 = lt::data_type::f32;
    const lt::dims d4 {B, H, S, D}, sc4 {B, H, S, S};
    const lt::dims in_st = direct ? lt::dims {S * stride, D, stride, 1} : lt::dims {H * S * D, S * D, D, 1};
    const lt::dims out_st = direct ? lt::dims {S * H * D, D, H * D, 1} : lt::dims {H * S * D, S * D, D, 1};
    size_t id = 0;
    lt q(id++, f16, d4, in_st), k(id++, f16, d4, in_st);
    lt scale(id++, f32, lt::dims {}, lt::layout_type::strided, lt::property_type::host_scalar);
    lt v(id++, f16, d4, in_st);
    lt score(id++, f32, sc4, lt::layout_type::strided), scaled(id++, f32, sc4, lt::layout_type::strided);
    lt probs(id++, f16, sc4, lt::layout_type::strided), out(id++, f16, d4, out_st);
    op bmm1(id++, op::kind::MatMul, {q, k}, {score}, "scores");
    bmm1.set_attr<bool>(op::attr::transpose_b, true);
    op mul(id++, op::kind::Multiply, {score, scale}, {scaled}, "scale");
    op sm(id++, op::kind::SoftMax, {scaled}, {probs}, "softmax");
    sm.set_attr<int64_t>(op::attr::axis, -1);
    sm.set_attr<std::string>(op::attr::mode, "inf_as_zero");
    op bmm2(id++, op::kind::MatMul, {probs, v}, {out}, "values");
    graph g(dnnl::engine::kind::gpu);
    g.add_op(bmm1); g.add_op(mul); g.add_op(sm); g.add_op(bmm2);
    g.finalize();
    auto parts = g.get_partitions();
    if (parts.size() != 1 || !parts[0].is_supported())
        throw std::runtime_error("oneDNN did not take the batched attention as one partition");
    const lt mine[4] = {q, k, scale, v};
    Ctx::Sdpa sd;
    for (const auto& port : parts[0].get_input_ports())
        for (int i = 0; i < 4; ++i)
            if (mine[i].get_id() == port.get_id()) { sd.in.push_back(mine[i]); sd.slot.push_back(i); }
    if (sd.in.size() != 4) throw std::runtime_error("oneDNN's batched attention partition has unexpected inputs");
    sd.cp = parts[0].compile(sd.in, {out}, c.eng);
    sd.out = sd.cp.query_logical_tensor(out.get_id());
    return c.sdpa_b.emplace(key, std::move(sd)).first->second;
}

// Returns false (having queued nothing) when this form does not apply or oneDNN refuses it.
static bool attention_fused_batch(Ctx& c, const void* q, const void* k, const void* v, int dt, int64_t B, int64_t S,
                                  int64_t H, int64_t D, int64_t stride, void* out, int out_dt) {
    // one chunk per sequence, as attention_fused would take it (a longer one keeps its chunking there)
    if (S > c.attn_rows) return false;
    sycl::queue& qu = c.q;
    c.sdpa_scale = 1.0f / std::sqrt((float) D);
    auto run = [&](Ctx::Sdpa& sd, const void* hq, const void* hk, const void* hv, void* ao) {
        const void* const handles[4] = {hq, hk, nullptr, hv};
        std::vector<dnnl::graph::tensor> in;
        for (size_t i = 0; i < sd.in.size(); ++i) {
            if (sd.slot[i] == 2) in.push_back(dnnl::graph::tensor::make_scalar_tensor(sd.in[i], &c.sdpa_scale));
            else in.emplace_back(sd.in[i], c.eng, const_cast<void*>(handles[sd.slot[i]]));
        }
        sd.cp.execute(c.strm, in, {dnnl::graph::tensor(sd.out, c.eng, ao)});
    };
    if (dt == NSD_F16 && out_dt == NSD_F16 && c.sdpa_direct_ok) {
        try {
            Ctx::Sdpa& sd = sdpa_batch_for(c, B, S, H, D, stride, true);
            run(sd, q, k, v, out);
            return true;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "h3sycl: batched attention in place refused (%s); copying by head instead\n", e.what());
            c.sdpa_direct_ok = false;
        }
    }
    Ctx::Sdpa* sd = nullptr;
    try {
        sd = &sdpa_batch_for(c, B, S, H, D, stride, false);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "h3sycl: batched attention refused (%s); one sequence at a time\n", e.what());
        return false;
    }
    const size_t n = (size_t) B * S * H * D;
    uint16_t* hq = c.grow(c.hq, c.hq_cap, n);
    uint16_t* hk = c.grow(c.hk, c.hk_cap, n);
    uint16_t* hv = c.grow(c.hv, c.hv_cap, n);
    float* aof = c.grow(c.ao, c.ao_cap, (n + 1) / 2);   // used as half [B, H, S, D]
    if (!hq || !hk || !hv || !aof) throw std::runtime_error(g_err);
    sycl::half* ao = (sycl::half*) aof;
    qu.parallel_for(sycl::range<3>((size_t) (B * H), (size_t) S, (size_t) D), [=](sycl::id<3> id) {
        const size_t b = id[0] / H, h = id[0] % H;
        const size_t src = (b * S + id[1]) * stride + h * D + id[2], dst = (id[0] * S + id[1]) * D + id[2];
        ((sycl::half*) hq)[dst] = (sycl::half) load(q, dt, src);
        ((sycl::half*) hk)[dst] = (sycl::half) load(k, dt, src);
        ((sycl::half*) hv)[dst] = (sycl::half) load(v, dt, src);
    });
    run(*sd, hq, hk, hv, ao);
    qu.parallel_for(sycl::range<3>((size_t) (B * S), (size_t) H, (size_t) D), [=](sycl::id<3> id) {
        const size_t b = id[0] / S, s = id[0] % S;
        store(out, out_dt, (id[0] * H + id[1]) * D + id[2], (float) ao[((b * H + id[1]) * S + s) * D + id[2]]);
    });
    return true;
}

// Form 0 (the default; NSD_ATTN=onednn skips it): SageAttention v1 - q and k quantized to int8 here, the attention by Intel's ARK kernel on
// sycl-tla in libh3sage.so (kernels/sage.cpp), v and the result in half. k's mean over the sequence is taken out
// before quantizing: it adds the same amount to every score of a row, which the softmax ignores, and what is left
// quantizes far better. One scale per head per kSageBlock rows. Whole sequence in one call: the kernel keeps its
// score tiles in registers, so memory does not grow with S^2.
constexpr int kSageBlock = 64;

static bool sage_load(Ctx& c) {
    if (c.sage_state) return c.sage_state > 0;
    c.sage_state = -1;
    std::string path = "libh3sage.so";
    Dl_info me;
    if (dladdr((void*) &sage_load, &me) && me.dli_fname) {     // beside this library
        std::string self = me.dli_fname;
        const size_t slash = self.rfind('/');
        if (slash != std::string::npos) path = self.substr(0, slash + 1) + path;
    }
    void* h = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h) { std::fprintf(stderr, "h3sycl: no SageAttention (%s); attention on oneDNN's kernel\n", dlerror()); return false; }
    c.sage_fn = (decltype(c.sage_fn)) dlsym(h, "h3sage_attention");
    c.sage_err = (decltype(c.sage_err)) dlsym(h, "h3sage_error");
    if (!c.sage_fn || !c.sage_err) { std::fprintf(stderr, "h3sycl: %s lacks h3sage_attention; attention stays on oneDNN\n", path.c_str()); return false; }
    std::fprintf(stderr, "h3sycl: attention: SageAttention v1 (int8 q, k) from %s\n", path.c_str());
    c.sage_state = 1;
    return true;
}

// Returns false (having queued nothing) when this shape is not one the kernel takes.
static bool attention_sage(Ctx& c, const void* q, const void* k, const void* v, int dt, int64_t S, int64_t H, int64_t D,
                           int64_t stride, void* out, int out_dt) {
    if ((D != 64 && D != 128) || S > INT32_MAX / D / H) return false;
    sycl::queue& qu = c.q;
    const size_t n = (size_t) S * H * D;
    const int64_t nb = (S + kSageBlock - 1) / kSageBlock;
    int8_t* iq = c.grow(c.sq, c.sq_cap, n);
    int8_t* ik = c.grow(c.sk, c.sk_cap, n);
    float* sc = c.grow(c.ss, c.ss_cap, (size_t) (2 * H * nb + H * D));
    uint16_t* hv = c.grow(c.hv, c.hv_cap, n);
    float* aof = c.grow(c.ao, c.ao_cap, (n + 1) / 2);                  // half [H, S, D]
    if (!iq || !ik || !sc || !hv || !aof) throw std::runtime_error(g_err);
    float* qs = sc;
    float* ks = sc + H * nb;
    float* kmean = sc + 2 * H * nb;
    sycl::half* ao = (sycl::half*) aof;

    const bool prof = c.profile;
    double t_ph[3] = {0, 0, 0};
    auto clock = std::chrono::steady_clock::now();
    auto lap = [&](int i) {
        if (!prof) return;
        qu.wait();
        const auto now = std::chrono::steady_clock::now();
        t_ph[i] += std::chrono::duration<double, std::milli>(now - clock).count();
        clock = now;
    };
    lap(0);
    // k's mean over the sequence, per head and channel
    qu.parallel_for(sycl::range<1>((size_t) (H * D)), [=](sycl::id<1> id) {
        float sum = 0.0f;
        for (int64_t t = 0; t < S; ++t) sum += load(k, dt, t * stride + id[0]);
        kmean[id[0]] = sum / (float) S;
    });
    // q and k - mean to int8: a work-group per (head, block of rows), the block's largest magnitude -> its scale
    constexpr int WG = 256;
    auto quant = [&](const void* src, int8_t* dst, float* scales, const float* mean) {
        qu.parallel_for(sycl::nd_range<1>((size_t) (H * nb * WG), WG), [=](sycl::nd_item<1> it) {
            const int64_t g = it.get_group(0), h = g / nb, b = g % nb;
            const int64_t r0 = b * kSageBlock, rows = sycl::min<int64_t>(kSageBlock, S - r0);
            const int lid = it.get_local_id(0);
            float m = 0.0f;
            for (int64_t i = lid; i < rows * D; i += WG) {
                const int64_t r = r0 + i / D, d = i % D;
                m = sycl::fmax(m, sycl::fabs(load(src, dt, r * stride + h * D + d) - (mean ? mean[h * D + d] : 0.0f)));
            }
            m = sycl::reduce_over_group(it.get_group(), m, sycl::maximum<float>());
            const float inv = m > 0.0f ? 127.0f / m : 0.0f;
            if (lid == 0) scales[h * nb + b] = m / 127.0f;
            for (int64_t i = lid; i < rows * D; i += WG) {
                const int64_t r = r0 + i / D, d = i % D;
                const float x = (load(src, dt, r * stride + h * D + d) - (mean ? mean[h * D + d] : 0.0f)) * inv;
                dst[(h * S + r) * D + d] = (int8_t) sycl::clamp(sycl::round(x), -127.0f, 127.0f);
            }
        });
    };
    quant(q, iq, qs, nullptr);
    quant(k, ik, ks, kmean);
    qu.parallel_for(sycl::range<3>((size_t) H, (size_t) S, (size_t) D), [=](sycl::id<3> id) {
        ((sycl::half*) hv)[(id[0] * S + id[1]) * D + id[2]] = (sycl::half) load(v, dt, id[1] * stride + id[0] * D + id[2]);
    });
    lap(0);
    if (c.sage_fn(&qu, iq, ik, hv, ao, qs, ks, kSageBlock, S, H, D, 1.0f / std::sqrt((float) D)) != 0)
        throw std::runtime_error(c.sage_err());
    lap(1);
    qu.parallel_for(sycl::range<3>((size_t) S, (size_t) H, (size_t) D), [=](sycl::id<3> id) {
        store(out, out_dt, (id[0] * H + id[1]) * D + id[2], (float) ao[(id[1] * S + id[0]) * D + id[2]]);
    });
    lap(2);
    if (prof)
        std::fprintf(stderr, "h3sycl: attention (sage) S=%lld: quantize+copies %.1f ms, attention %.1f, out %.1f\n",
                     (long long) S, t_ph[0], t_ph[1], t_ph[2]);
    return true;
}

int nsd_attention(void* ctx, const void* q, const void* k, const void* v, int dt, int64_t S, int64_t H, int64_t D,
                  int64_t stride, void* out, int out_dt) try {
    auto& c = *static_cast<Ctx*>(ctx);
    if (S <= 0 || H <= 0 || D <= 0) return 0;
    if (stride < H * D) { g_err = "nsd_attention: the row stride is shorter than a row"; return -1; }
    if (c.sage_want && S >= c.sage_min_s && sage_load(c)) {
        try {
            if (attention_sage(c, q, k, v, dt, S, H, D, stride, out, out_dt)) return 0;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "h3sycl: SageAttention failed (%s); attention stays on oneDNN\n", e.what());
            c.q.wait();
            c.sage_state = -1;
        }
    }
    if (c.sdpa_ok) {
        if (attention_fused(c, q, k, v, dt, S, H, D, stride, out, out_dt)) return 0;
        c.sdpa_ok = false;
    }
    return attention_split(c, q, k, v, dt, S, H, D, stride, out, out_dt);
} catch (const std::exception& e) { g_err = e.what(); return -1; }

int nsd_attention_batch(void* ctx, const void* q, const void* k, const void* v, int dt, int64_t B, int64_t S, int64_t H,
                        int64_t D, int64_t stride, void* out, int out_dt) try {
    auto& c = *static_cast<Ctx*>(ctx);
    if (B <= 0 || S <= 0 || H <= 0 || D <= 0) return 0;
    if (stride < H * D) { g_err = "nsd_attention_batch: the row stride is shorter than a row"; return -1; }
    // SageAttention takes long sequences one at a time; short ones go to oneDNN's fused kernel, here all at once
    const bool sage = c.sage_want && S >= c.sage_min_s;
    if (B > 1 && !sage && c.sdpa_ok && attention_fused_batch(c, q, k, v, dt, B, S, H, D, stride, out, out_dt)) return 0;
    const size_t ei = dt == NSD_F32 ? 4 : 2, eo = out_dt == NSD_F32 ? 4 : 2;
    for (int64_t b = 0; b < B; ++b) {
        const size_t io = (size_t) (b * S * stride) * ei, oo = (size_t) (b * S * H * D) * eo;
        const int rc = nsd_attention(ctx, (const char*) q + io, (const char*) k + io, (const char*) v + io, dt, S, H, D, stride,
                                     (char*) out + oo, out_dt);
        if (rc != 0) return rc;
    }
    return 0;
} catch (const std::exception& e) { g_err = e.what(); return -1; }

static int attention_split(Ctx& c, const void* q, const void* k, const void* v, int dt, int64_t S, int64_t H, int64_t D,
                           int64_t stride, void* out, int out_dt) {
    using dnnl::memory;
    sycl::queue& qu = c.q;
    const int64_t D1 = D + 1;                           // v with its column of ones
    // rows per chunk: the scores of a chunk take at most ~1.5 GiB
    const int64_t rows_max = std::max<int64_t>(1, std::min<int64_t>(S, (int64_t) c.attn_table_bytes / (H * S * 2)));
    // the max pass: work-item j of a row takes scores j, j + kLanes, j + 2 kLanes ..., so neighbouring work-items
    // (which the GPU runs as the lanes of one vector) read neighbouring scores
    constexpr int64_t kLanes = 256;
    const int64_t G = kLanes;
    uint16_t* hq = c.grow(c.hq, c.hq_cap, (size_t) H * rows_max * D);
    uint16_t* hk = c.grow(c.hk, c.hk_cap, (size_t) S * H * D);
    uint16_t* hv = c.grow(c.hv, c.hv_cap, (size_t) S * H * D1);
    uint16_t* sc = c.grow(c.sc, c.sc_cap, (size_t) H * rows_max * S);
    float* ao = c.grow(c.ao, c.ao_cap, (size_t) H * rows_max * D1);
    float* part = c.grow(c.part, c.part_cap, (size_t) (H * rows_max * G));
    float* rowv = c.grow(c.rowv, c.rowv_cap, (size_t) (H * rows_max));
    if (!hq || !hk || !hv || !sc || !ao || !part || !rowv) return -1;

    const bool prof = c.profile;
    double t_ph[6] = {0, 0, 0, 0, 0, 0};
    auto clock = std::chrono::steady_clock::now();
    auto lap = [&](int i) {
        if (!prof) return;
        qu.wait();
        const auto now = std::chrono::steady_clock::now();
        t_ph[i] += std::chrono::duration<double, std::milli>(now - clock).count();
        clock = now;
    };
    lap(0);
    qu.parallel_for(sycl::range<3>((size_t) H, (size_t) S, (size_t) D1), [=](sycl::id<3> id) {
        const size_t h = id[0], t = id[1], d = id[2];
        if ((int64_t) d < D) {
            const size_t src = t * stride + h * D + d;
            ((sycl::half*) hk)[(h * D + d) * S + t] = (sycl::half) load(k, dt, src);
            ((sycl::half*) hv)[(h * S + t) * D1 + d] = (sycl::half) load(v, dt, src);
        } else {
            ((sycl::half*) hv)[(h * S + t) * D1 + d] = (sycl::half) 1.0f;
        }
    });
    const float scale = 1.0f / std::sqrt((float) D);
    const auto f16 = memory::data_type::f16;
    for (int64_t r0 = 0; r0 < S; r0 += rows_max) {
        const int64_t rows = std::min(rows_max, S - r0);
        const size_t R = (size_t) (H * rows);           // rows of the score table of this chunk
        qu.parallel_for(sycl::range<3>((size_t) H, (size_t) rows, (size_t) D), [=](sycl::id<3> id) {
            ((sycl::half*) hq)[(id[0] * rows + id[1]) * D + id[2]] =
                    (sycl::half) (load(q, dt, (r0 + id[1]) * stride + id[0] * D + id[2]) * scale);
        });
        memory::desc q_md({H, rows, D}, f16, memory::format_tag::abc);
        memory::desc k_md({H, D, S}, f16, memory::format_tag::abc);
        memory::desc s_md({H, rows, S}, f16, memory::format_tag::abc);
        memory::desc v_md({H, S, D1}, f16, memory::format_tag::abc);
        memory::desc o_md({H, rows, D1}, memory::data_type::f32, memory::format_tag::abc);
        auto key = std::make_tuple(H, rows, S, D);
        auto it = c.attn.find(key);
        if (it == c.attn.end()) {
            Ctx::Attn a{dnnl::matmul(dnnl::matmul::primitive_desc(c.eng, q_md, k_md, s_md)),
                        dnnl::matmul(dnnl::matmul::primitive_desc(c.eng, s_md, v_md, o_md))};
            it = c.attn.emplace(key, std::move(a)).first;
        }
        auto s_mem = usm(s_md, c.eng, sc);
        lap(0);
        it->second.qk.execute(c.strm, {{DNNL_ARG_SRC, usm(q_md, c.eng, hq)}, {DNNL_ARG_WEIGHTS, usm(k_md, c.eng, hk)}, {DNNL_ARG_DST, s_mem}});
        lap(1);
        // the row's max (read only)
        sycl::half* sh = (sycl::half*) sc;
        qu.parallel_for(sycl::range<2>(R, (size_t) G), [=](sycl::id<2> id) {
            const size_t base = id[0] * S;
            float m = -3.0e38f;
            for (size_t i = id[1]; i < (size_t) S; i += kLanes) m = sycl::fmax(m, (float) sh[base + i]);
            part[id[0] * G + id[1]] = m;
        });
        qu.parallel_for(sycl::range<1>(R), [=](sycl::id<1> r) {
            float m = -3.0e38f;
            for (int64_t g = 0; g < G; ++g) m = sycl::fmax(m, part[r[0] * G + g]);
            rowv[r[0]] = m;
        });
        lap(2);
        // weights in place, each at most 1
        qu.parallel_for(sycl::range<2>(R, (size_t) S), [=](sycl::id<2> id) {
            const size_t i = id[0] * S + id[1];
            sh[i] = (sycl::half) exp_neg((float) sh[i] - rowv[id[0]]);
        });
        lap(3);
        it->second.pv.execute(c.strm, {{DNNL_ARG_SRC, s_mem}, {DNNL_ARG_WEIGHTS, usm(v_md, c.eng, hv)}, {DNNL_ARG_DST, usm(o_md, c.eng, ao)}});
        lap(4);
        // divide by the row's sum of weights (the last column), and back to token order: out [S, H * D]
        qu.parallel_for(sycl::range<3>((size_t) rows, (size_t) H, (size_t) D), [=](sycl::id<3> id) {
            const size_t hr = (id[1] * rows + id[0]) * D1;
            store(out, out_dt, ((r0 + id[0]) * H + id[1]) * D + id[2], ao[hr + id[2]] / ao[hr + D]);
        });
        lap(5);
    }
    if (prof)
        std::fprintf(stderr, "h3sycl: attention (split) S=%lld: copies %.1f ms, scores %.1f, row max %.1f, exp %.1f, values %.1f, out %.1f\n",
                     (long long) S, t_ph[0], t_ph[1], t_ph[2], t_ph[3], t_ph[4], t_ph[5]);
    return 0;
}

}  // extern "C"

extern "C" int nsd_attention_qk(void* ctx, const void* q, int64_t Sq, int64_t qs, const void* k, const void* v, int64_t Skv,
                                int64_t kvs, int dt, int64_t H, int64_t D, void* out, int out_dt) try {
    auto& c = *static_cast<Ctx*>(ctx);
    if (Sq <= 0 || Skv <= 0) return 0;
    attention_fused_qk(c, q, Sq, qs, k, v, Skv, kvs, dt, H, D, out, out_dt);
    return 0;
} catch (const std::exception& e) { g_err = e.what(); return -1; }
