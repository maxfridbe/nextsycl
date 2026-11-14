// gguf-min.hpp: just enough of a GGUF file to find a tensor (its type, shape and bytes) for the reference programs.
// Header-only; reads the key/value section only to step over it.
#pragma once
#include <cstdint>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

struct GgufTensor { std::string name; uint32_t type = 0; std::vector<uint64_t> dims; uint64_t offset = 0; };

struct Gguf {
    std::string path;
    std::map<std::string, GgufTensor> tensors;
    uint64_t data = 0;   // the data section's offset in the file

    explicit Gguf(const std::string& p) : path(p) {
        FILE* f = fopen(p.c_str(), "rb");
        if (!f) throw std::runtime_error("open " + p);
        auto rd = [&](void* d, size_t n) { if (fread(d, 1, n, f) != n) throw std::runtime_error("short read " + p); };
        auto u32 = [&] { uint32_t v; rd(&v, 4); return v; };
        auto u64 = [&] { uint64_t v; rd(&v, 8); return v; };
        auto str = [&] { std::string s(u64(), '\0'); rd(s.data(), s.size()); return s; };
        if (u32() != 0x46554747u) throw std::runtime_error("not GGUF: " + p);
        u32();
        const uint64_t nt = u64(), nkv = u64();
        uint64_t align = 32;
        const int fixed[13] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
        for (uint64_t i = 0; i < nkv; ++i) {
            const std::string key = str();
            const uint32_t t = u32();
            if (t == 8) { str(); continue; }
            if (t == 9) {
                const uint32_t at = u32(); const uint64_t n = u64();
                if (at == 8) for (uint64_t k = 0; k < n; ++k) str();
                else fseeko(f, (off_t) (fixed[at] * n), SEEK_CUR);
                continue;
            }
            if (key == "general.alignment" && t == 4) { align = u32(); continue; }
            fseeko(f, fixed[t], SEEK_CUR);
        }
        for (uint64_t i = 0; i < nt; ++i) {
            GgufTensor x;
            x.name = str();
            const uint32_t nd = u32();
            for (uint32_t k = 0; k < nd; ++k) x.dims.push_back(u64());
            x.type = u32();
            x.offset = u64();
            tensors[x.name] = x;
        }
        const uint64_t pos = (uint64_t) ftello(f);
        data = (pos + align - 1) / align * align;
        fclose(f);
    }
    // `bytes` bytes of tensor t from byte `at` of it
    std::vector<uint8_t> read(const GgufTensor& t, uint64_t at, uint64_t bytes) const {
        std::vector<uint8_t> b(bytes);
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) throw std::runtime_error("open " + path);
        fseeko(f, (off_t) (data + t.offset + at), SEEK_SET);
        const size_t got = fread(b.data(), 1, bytes, f);
        fclose(f);
        if (got != bytes) throw std::runtime_error("short read of " + t.name);
        return b;
    }
};

// bytes of n values of a ggml type (the types the decode kernels take); 0 when unknown
inline uint64_t ggml_bytes(uint32_t t, uint64_t n) {
    switch (t) {
        case 0: return n * 4;
        case 1: case 30: return n * 2;
        case 2: return n / 32 * 18;    case 3: return n / 32 * 20;    case 6: return n / 32 * 22;
        case 7: return n / 32 * 24;    case 8: return n / 32 * 34;    case 20: return n / 32 * 18;
        case 10: return n / 256 * 84;  case 11: return n / 256 * 110; case 12: return n / 256 * 144;
        case 13: return n / 256 * 176; case 14: return n / 256 * 210; case 16: return n / 256 * 66;
        case 17: return n / 256 * 74;  case 18: return n / 256 * 98;  case 19: return n / 256 * 50;
        case 21: return n / 256 * 110; case 22: return n / 256 * 82;  case 23: return n / 256 * 136;
        case 29: return n / 256 * 56;  case 39: return n / 32 * 17;   case 42: return n / 64 * 18;
        default: return 0;
    }
}
