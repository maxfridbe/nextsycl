// xq.hpp: the museglimmer silo's decode-activation layout (silo.h, ns_mg_xq), shared by its products: block b's quants
// at b x 512 bytes - word w (quants 4w .. 4w + 3) of column c at w x 64 + c x 4: the packed int8 B operand - then the
// scales [n_in / 32][16] float, then the blocks' sums [n_in / 32][16] float (scale x the sum of its quants: a K-quant's
// minimum against them), then the quants again token-major (the A operand of the K-quant products: block b's at b x 512,
// column c's 32 at c x 32)
#ifndef NS_MG_XQ_HPP
#define NS_MG_XQ_HPP
#include <cstddef>
#include <cstdint>

namespace mgxq {
constexpr int64_t XCOLS = 16;
inline size_t align256(size_t n) { return (n + 255) / 256 * 256; }
// where column c's element i goes, where the scales and the sums start
inline int64_t at(int64_t c, int64_t i) { return (i / 32) * 512 + ((i % 32) / 4) * 64 + c * 4 + (i % 4); }
inline size_t scales(int64_t n_in) { return align256((size_t) (n_in / 32) * 512); }
inline size_t sums(int64_t n_in) { return scales(n_in) + (size_t) (XCOLS * (n_in / 32)) * 4; }
inline size_t rowq(int64_t n_in) { return align256(sums(n_in) + (size_t) (XCOLS * (n_in / 32)) * 4); }
inline int64_t at_a(int64_t c, int64_t i) { return (i / 32) * 512 + c * 32 + (i % 32); }
inline size_t bytes(int64_t n_in) { return rowq(n_in) + (size_t) (n_in / 32) * 512; }
}  // namespace mgxq
#endif
