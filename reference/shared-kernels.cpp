// Build in the oneAPI image: icpx -O2 -fsycl -Ikernels/ns reference/shared-kernels.cpp -o shared_kernels dist/libnextsycl-llm.so -Wl,-rpath,dist
// The shared decode kernels (kernels/llm/common: ns_mmvq, ns_dequant_f16, ns_moe_grouped) against Strata's (ns_ref_*,
// while both are in the library) on real weights from GGUF files: their outputs compared, and their rates - "hot" one
// matrix rerun (mostly from L2), "cold" copies of it cycled through (512 MB in all, as a token reads each once).
// `shared_kernels GPU LIST`, a line of LIST one test:
//   mm  FILE TENSOR [EXPERT]                    a decode product (1 and 4 columns, random activations)
//   dq  FILE TENSOR [EXPERT]                    the expansion to fp16 (the first 256 x 1024 values or the matrix)
//   moe FILE GATE FILE UP FILE DOWN TOKENS USED [LIMIT [LANES]]   a layer's grouped experts: TOKENS tokens, USED experts each
//                                               (LANES a row, as the engine passes them)
#include "ns.h"
#include "gguf-min.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

extern "C" {
size_t ns_q8_1_bytes(int64_t n_in, int64_t ncols);
int ns_quantize_q8_1(ns_gpu* g, const float* x, void* q8_1, int64_t n_in, int64_t ncols);
int ns_mmvq(ns_gpu* g, int type, const void* w, const void* x_q8_1, float* y, int64_t n_in, int64_t n_out, int64_t ncols);
int ns_ref_mmvq(ns_gpu* g, int type, const void* w, const void* x_q8_1, float* y, int64_t n_in, int64_t n_out, int64_t ncols);
int ns_ref_quantize_q8_1(ns_gpu* g, const float* x, void* q8_1, int64_t n_in, int64_t ncols);
int ns_dequant_f16(ns_gpu* g, int type, const void* src, int64_t n, uint16_t* dst);
int ns_ref_dequant_f16(ns_gpu* g, int type, const void* src, int64_t n, uint16_t* dst);
size_t ns_moe_scratch_bytes(int64_t entries, int64_t n_ff);
size_t ns_ref_moe_scratch_bytes(int64_t entries, int64_t n_ff);
typedef int (*moe_fn)(ns_gpu*, int, int, int64_t, int64_t, const uint64_t*, const int32_t*, const int32_t*, const int32_t*, const int32_t*, int64_t, int64_t,
                      const void*, void*, float*, float, int, int);
int ns_moe_grouped(ns_gpu* g, int gu_type, int d_type, int64_t n_embd, int64_t n_ff, const uint64_t* grp_ptr, const int32_t* grp_start,
                   const int32_t* n_groups, const int32_t* ent_dst, const int32_t* ent_tok, int64_t groups, int64_t entries, const void* x_q8_1,
                   void* scratch, float* out, float limit, int lanes, int phase);
int ns_ref_moe_grouped(ns_gpu* g, int gu_type, int d_type, int64_t n_embd, int64_t n_ff, const uint64_t* grp_ptr, const int32_t* grp_start,
                       const int32_t* n_groups, const int32_t* ent_dst, const int32_t* ent_tok, int64_t groups, int64_t entries, const void* x_q8_1,
                       void* scratch, float* out, float limit, int lanes, int phase);
}

static ns_gpu* G;
static void ck(int rc, const char* what) { if (rc) { fprintf(stderr, "%s: %s\n", what, ns_last_error()); exit(1); } }
static void* dev(size_t b) { void* p; ck(ns_alloc(G, b, &p), "alloc"); return p; }
static void* up(const void* h, size_t b) { void* p = dev(b); ck(ns_copy_to(G, p, h, b), "copy"); ck(ns_sync(G), "sync"); return p; }
template <typename T> static std::vector<T> down(const void* d, size_t n) { std::vector<T> v(n); ck(ns_copy_from(G, v.data(), d, n * sizeof(T)), "copy"); ck(ns_sync(G), "sync"); return v; }
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static const char* tname(uint32_t t) {
    switch (t) {
        case 2: return "Q4_0"; case 6: return "Q5_0"; case 8: return "Q8_0"; case 10: return "Q2_K"; case 11: return "Q3_K";
        case 12: return "Q4_K"; case 13: return "Q5_K"; case 14: return "Q6_K"; case 16: return "IQ2_XXS"; case 17: return "IQ2_XS";
        case 18: return "IQ3_XXS"; case 19: return "IQ1_S"; case 20: return "IQ4_NL"; case 21: return "IQ3_S"; case 22: return "IQ2_S";
        case 23: return "IQ4_XS"; case 29: return "IQ1_M"; default: return "?";
    }
}
// largest |a - b| over largest |b|, and how many differ at all
static void cmp(const std::vector<float>& a, const std::vector<float>& b, double& rel, size_t& ndiff) {
    double m = 0, r = 0; ndiff = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        m = std::max(m, (double) std::fabs(a[i] - b[i])); r = std::max(r, (double) std::fabs(b[i]));
        if (std::memcmp(&a[i], &b[i], 4) != 0) ++ndiff;
    }
    rel = r > 0 ? m / r : m;
}
static float h2f(uint16_t h) {
    const uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
    float v = e == 0 ? std::ldexp((float) m, -24) : e == 31 ? (m ? NAN : INFINITY) : std::ldexp((float) (m | 1024), (int) e - 25);
    return s ? -v : v;
}

// one matrix (or expert) of a tensor: its bytes, rows (outputs), cols (inputs)
struct Mat { uint32_t type; int64_t rows, cols; std::vector<uint8_t> b; std::string name; };
static Mat matrix(const Gguf& f, const std::string& name, int expert) {
    auto it = f.tensors.find(name);
    if (it == f.tensors.end()) { fprintf(stderr, "no tensor %s\n", name.c_str()); exit(1); }
    const GgufTensor& t = it->second;
    Mat m; m.type = t.type; m.cols = (int64_t) t.dims[0]; m.rows = (int64_t) t.dims[1]; m.name = name;
    const uint64_t bytes = ggml_bytes(t.type, (uint64_t) (m.rows * m.cols));
    m.b = f.read(t, t.dims.size() > 2 ? bytes * (uint64_t) std::max(expert, 0) : 0, bytes);
    return m;
}

typedef int (*mm_fn)(ns_gpu*, int, const void*, const void*, float*, int64_t, int64_t, int64_t);
static void rates(mm_fn fn, const Mat& m, void* w, const std::vector<void*>& copies, void* q, float* y, double& hot, double& cold) {
    const double bytes = (double) m.b.size();
    for (int i = 0; i < 3; ++i) ck(fn(G, (int) m.type, w, q, y, m.cols, m.rows, 1), "mmvq");
    ck(ns_sync(G), "sync");
    const int n = 50;
    double t0 = now();
    for (int i = 0; i < n; ++i) ck(fn(G, (int) m.type, w, q, y, m.cols, m.rows, 1), "mmvq");
    ck(ns_sync(G), "sync");
    hot = bytes * n / (now() - t0) / 1e9;
    t0 = now();
    for (void* c : copies) ck(fn(G, (int) m.type, c, q, y, m.cols, m.rows, 1), "mmvq");
    ck(ns_sync(G), "sync");
    cold = bytes * (double) copies.size() / (now() - t0) / 1e9;
}

static void test_mm(const Gguf& f, const std::string& name, int expert) {
    const Mat m = matrix(f, name, expert);
    void* w = up(m.b.data(), m.b.size());
    std::vector<void*> copies;
    for (size_t k = 0; k < std::max<size_t>(2, (512u << 20) / m.b.size()); ++k) copies.push_back(up(m.b.data(), m.b.size()));
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> xs(4 * m.cols);
    for (auto& v : xs) v = nd(rng);
    float* x = (float*) up(xs.data(), xs.size() * 4);
    void* q = dev(ns_q8_1_bytes(m.cols, 4));
    void* qr = dev(ns_q8_1_bytes(m.cols, 4));
    float* y = (float*) dev(4 * m.rows * 4);
    float* yr = (float*) dev(4 * m.rows * 4);
    float* y1 = (float*) dev(m.rows * 4);
    ck(ns_quantize_q8_1(G, x, q, m.cols, 4), "quantize");
    ck(ns_ref_quantize_q8_1(G, x, qr, m.cols, 4), "ref quantize");
    const auto qa = down<uint8_t>(q, ns_q8_1_bytes(m.cols, 4)), qb = down<uint8_t>(qr, ns_q8_1_bytes(m.cols, 4));
    const bool qsame = qa == qb;
    ck(ns_mmvq(G, (int) m.type, w, q, y, m.cols, m.rows, 4), "mmvq");
    ck(ns_ref_mmvq(G, (int) m.type, w, q, yr, m.cols, m.rows, 4), "ref mmvq");
    ck(ns_mmvq(G, (int) m.type, w, q, y1, m.cols, m.rows, 1), "mmvq 1");
    const auto a = down<float>(y, 4 * m.rows), b = down<float>(yr, 4 * m.rows), a1 = down<float>(y1, m.rows);
    double rel; size_t nd4;
    cmp(a, b, rel, nd4);
    const bool col_exact = std::memcmp(a.data(), a1.data(), m.rows * 4) == 0;
    // both against the exact product of column 0 (the weights expanded to half - Strata's expansion where it takes the
    // type, else ours - and the float activations): the largest error over the largest |y|
    double eo = 0, er = 0, ym = 0;
    {
        const int64_t n = m.rows * m.cols;
        uint16_t* wh = (uint16_t*) dev(n * 2);
        if (ns_ref_dequant_f16(G, (int) m.type, w, n, wh)) ck(ns_dequant_f16(G, (int) m.type, w, n, wh), "dequant");
        const auto h = down<uint16_t>(wh, n);
        ns_free(G, wh);
        for (int64_t r = 0; r < m.rows; ++r) {
            double t = 0;
            for (int64_t c = 0; c < m.cols; ++c) t += (double) h2f(h[r * m.cols + c]) * xs[c];
            eo = std::max(eo, std::fabs(a[r] - t)); er = std::max(er, std::fabs(b[r] - t)); ym = std::max(ym, std::fabs(t));
        }
    }
    double oh, oc, rh, rc;
    rates(ns_mmvq, m, w, copies, q, y1, oh, oc);
    rates(ns_ref_mmvq, m, w, copies, q, y1, rh, rc);
    printf("mm  %-8s %-34s %6lldx%-6lld q8_1 %s  own-strata %.1e (%zu/%lld differ)  vs exact: own %.1e strata %.1e  1col=4col[0] %s  own %4.0f hot %4.0f cold  strata %4.0f hot %4.0f cold GB/s  %s\n",
           tname(m.type), name.c_str(), (long long) m.rows, (long long) m.cols, qsame ? "same" : "DIFF", rel, nd4, (long long) (4 * m.rows), eo / ym, er / ym,
           col_exact ? "yes" : "NO", oh, oc, rh, rc, oc >= rc * 0.98 ? "ok" : "SLOWER");
    for (void* c : copies) ns_free(G, c);
    for (void* p : {w, (void*) x, q, qr, (void*) y, (void*) yr, (void*) y1}) ns_free(G, p);
}

static void test_dq(const Gguf& f, const std::string& name, int expert) {
    const Mat m = matrix(f, name, expert);
    const int64_t n = m.rows * m.cols;
    void* w = up(m.b.data(), m.b.size());
    uint16_t* a = (uint16_t*) dev(n * 2);
    uint16_t* b = (uint16_t*) dev(n * 2);
    ck(ns_dequant_f16(G, (int) m.type, w, n, a), "dequant");
    ck(ns_ref_dequant_f16(G, (int) m.type, w, n, b), "ref dequant");
    const auto va = down<uint16_t>(a, n), vb = down<uint16_t>(b, n);
    size_t diff = 0; double md = 0, mr = 0;
    for (int64_t i = 0; i < n; ++i) {
        if (va[i] != vb[i]) { ++diff; md = std::max(md, (double) std::fabs(h2f(va[i]) - h2f(vb[i]))); }
        mr = std::max(mr, (double) std::fabs(h2f(vb[i])));
    }
    auto time = [&](int (*fn)(ns_gpu*, int, const void*, int64_t, uint16_t*), uint16_t* dst) {
        ck(fn(G, (int) m.type, w, n, dst), "dq");
        ck(ns_sync(G), "sync");
        const double t0 = now();
        for (int i = 0; i < 20; ++i) ck(fn(G, (int) m.type, w, n, dst), "dq");
        ck(ns_sync(G), "sync");
        return (now() - t0) / 20 * 1e6;
    };
    const double to = time(ns_dequant_f16, a), tr = time(ns_ref_dequant_f16, b);
    printf("dq  %-8s %-40s %6lldx%-6lld %zu/%lld halves differ (max %.2e of %.2e)  own %7.1f us  strata %7.1f us  %s\n", tname(m.type), name.c_str(),
           (long long) m.rows, (long long) m.cols, diff, (long long) n, md, mr, to, tr, to <= tr * 1.02 ? "ok" : "SLOWER");
    ns_free(G, w); ns_free(G, a); ns_free(G, b);
}

static void test_moe(const Gguf& f, const std::string& gname, const Gguf& fu, const std::string& uname, const Gguf& fd, const std::string& dname, int tokens,
                     int used, float limit, int lanes) {
    const GgufTensor& gt = f.tensors.at(gname);
    const int64_t n_embd = (int64_t) gt.dims[0], n_ff = (int64_t) gt.dims[1], n_exp = (int64_t) gt.dims[2];
    const uint32_t gu_type = gt.type, d_type = fd.tensors.at(dname).type;
    // `used` distinct experts (spread over the layer), each a blob [gate | up | down]
    std::vector<int> ex(used);
    for (int k = 0; k < used; ++k) ex[k] = (int) ((int64_t) k * n_exp / used + 3) % (int) n_exp;
    std::vector<void*> blobs;
    std::vector<uint64_t> ptr;
    size_t blob_bytes = 0;
    for (int e : ex) {
        const Mat a = matrix(f, gname, e), b = matrix(fu, uname, e), c = matrix(fd, dname, e);
        std::vector<uint8_t> blob(a.b);
        blob.insert(blob.end(), b.b.begin(), b.b.end());
        blob.insert(blob.end(), c.b.begin(), c.b.end());
        blob_bytes = blob.size();
        blobs.push_back(up(blob.data(), blob.size()));
        ptr.push_back((uint64_t) blobs.back());
    }
    // every token routed to every one of the experts: group k = expert k, entries (k, t) for each token t
    const int groups = used, entries = used * tokens;
    std::vector<int32_t> start(groups + 1), dst(entries), tok(entries);
    for (int k = 0; k <= groups; ++k) start[k] = k * tokens;
    for (int k = 0; k < groups; ++k) for (int t = 0; t < tokens; ++t) { dst[k * tokens + t] = k * tokens + t; tok[k * tokens + t] = t; }
    const int32_t ng = groups;
    uint64_t* dptr = (uint64_t*) up(ptr.data(), ptr.size() * 8);
    int32_t* dstart = (int32_t*) up(start.data(), start.size() * 4);
    int32_t* dng = (int32_t*) up(&ng, 4);
    int32_t* ddst = (int32_t*) up(dst.data(), dst.size() * 4);
    int32_t* dtok = (int32_t*) up(tok.data(), tok.size() * 4);
    std::mt19937 rng(99);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> xs(tokens * n_embd);
    for (auto& v : xs) v = nd(rng);
    float* x = (float*) up(xs.data(), xs.size() * 4);
    void* q = dev(ns_q8_1_bytes(n_embd, tokens));
    ck(ns_quantize_q8_1(G, x, q, n_embd, tokens), "quantize");
    const size_t sb = std::max(ns_moe_scratch_bytes(entries, n_ff), ns_ref_moe_scratch_bytes(entries, n_ff));
    void* scr = dev(sb);
    float* oa = (float*) dev(entries * n_embd * 4);
    float* ob = (float*) dev(entries * n_embd * 4);
    auto run = [&](moe_fn fn, float* out, int phase = 0) {
        ck(fn(G, (int) gu_type, (int) d_type, n_embd, n_ff, dptr, dstart, dng, ddst, dtok, groups, entries, q, scr, out, limit, lanes, phase), "moe");
    };
    run(ns_moe_grouped, oa);
    run(ns_ref_moe_grouped, ob);
    const auto a = down<float>(oa, entries * n_embd), b = down<float>(ob, entries * n_embd);
    double rel; size_t ndf;
    cmp(a, b, rel, ndf);
    auto time = [&](moe_fn fn, float* out, int phase = 0) {
        run(fn, out, phase);
        ck(ns_sync(G), "sync");
        const double t0 = now();
        for (int i = 0; i < 20; ++i) run(fn, out, phase);
        ck(ns_sync(G), "sync");
        return (now() - t0) / 20 * 1e6;
    };
    const double to = time(ns_moe_grouped, oa), tr = time(ns_ref_moe_grouped, ob);
    const double o1 = time(ns_moe_grouped, oa, 1), r1 = time(ns_ref_moe_grouped, ob, 1);
    printf("moe %-8s/%-8s %4lld x %-5lld lanes %2d tokens %d x %d experts  err %.2e (%zu/%d differ)  own %7.1f us (gate|up %6.1f)  strata %7.1f us (gate|up %6.1f)  %s\n",
           tname(gu_type), tname(d_type), (long long) n_embd, (long long) n_ff, lanes, tokens, used, rel, ndf, entries * (int) n_embd, to, o1,
           tr, r1, to <= tr * 1.02 ? "ok" : "SLOWER");
    for (void* p : blobs) ns_free(G, p);
    for (void* p : {(void*) dptr, (void*) dstart, (void*) dng, (void*) ddst, (void*) dtok, (void*) x, q, scr, (void*) oa, (void*) ob}) ns_free(G, p);
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "shared_kernels GPU LIST\n"); return 2; }
    ck(ns_gpu_open(atoi(argv[1]), &G), "open");
    char name[128] = {};
    ns_gpu_name(atoi(argv[1]), name, sizeof name);
    printf("gpu %s: %s\n", argv[1], name);
    std::ifstream in(argv[2]);
    std::string line;
    std::map<std::string, Gguf*> files;
    auto file = [&](const std::string& p) -> Gguf& { if (!files.count(p)) files[p] = new Gguf(p); return *files[p]; };
    while (std::getline(in, line)) {
        std::istringstream s(line);
        std::string op, path, a, b, c;
        if (!(s >> op) || op[0] == '#') continue;
        s >> path;
        if (op == "mm" || op == "dq") {
            int e = -1;
            s >> a >> e;
            if (op == "mm") test_mm(file(path), a, e); else test_dq(file(path), a, e);
        } else if (op == "moe") {
            std::string pu, pd;
            int tokens = 1, used = 8, lanes = 0; float limit = 0;
            s >> a >> pu >> b >> pd >> c >> tokens >> used >> limit >> lanes;
            test_moe(file(path), a, file(pu), b, file(pd), c, tokens, used, limit, lanes);
        }
        fflush(stdout);
    }
    return 0;
}
