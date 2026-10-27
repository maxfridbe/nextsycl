/* silo.h: the MOSS-Image engine's silo (dist/silo/libnextsycl-mossimage.so): the DiT's per-block passes fused and read as
 * streams of 16-byte vectors - the engine (image/mossimage/src/ffi.rs) uses them when the library is there and its own
 * kernels (mo.h, the shared nsd ones) otherwise. Every call is queued on the GPU's queue and returns 0, or -1 with the
 * reason in ns_last_error(). */
#ifndef NS_IMAGE_MOSS_SILO_H
#define NS_IMAGE_MOSS_SILO_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* Whether this GPU runs the passes below at these sizes (1) or the engine must use its own kernels (0): sub-groups of 16, C
 * (the row) a multiple of 1024 up to 4096, D (the head) 128 */
int ns_image_moss_silo_supported(ns_gpu* g, int64_t C, int64_t D);

/* x float32 [M, C] (C a multiple of 8, a power-of-two-friendly 3072 here): optionally x += resid * gate first (resid
 * float32 [M, C], gate float32 [C], one row for every row; written back to x), then out half = x / rms(x) * weight *
 * (1 + scale) + shift (scale and shift float32 [C], the same row for every row; NULL: no modulation). One pass over
 * x and resid where gate_add + rms_norm_mod took two and three. */
int ns_image_moss_silo_add_norm(ns_gpu* g, float* x, const float* resid, const float* gate, int64_t M, int64_t C, const float* weight, float eps,
                                const float* scale, const float* shift, void* out);

/* The q and k heads of the packed qkv rows (half, row stride `stride`, q at column 0, k at column `kcol`): each head's
 * RMS norm (weight wq / wk float32 [D], eps) and the rotary phases cs float32 [M, D] (cos, sin pairs; pair j of a
 * token at cs[token * D + 2 j]) applied half-split ((j, D / 2 + j) pairs) - in place, both in one launch. D = 128. */
int ns_image_moss_silo_qk_rope(ns_gpu* g, void* qkv, int64_t M, int64_t H, int64_t D, int64_t stride, int64_t kcol, const float* wq, const float* wk,
                               float eps, const float* cs);

/* The DiT's gated activation (mo.h's situ) from half rows: out[r, c] = 4 tanh(gate / 4) sigmoid(gate) * 25 tanh(hid / 25),
 * hid and gate half rows `ldh` apart (a fused up/gate matrix's two halves: gate = hid + C), out rows `ldo` apart.
 * C a multiple of 8. */
int ns_image_moss_silo_situ(ns_gpu* g, const void* hid, const void* gate, int64_t ldh, int64_t M, int64_t C, void* out, int64_t ldo);

/* Causal attention for the text encoder (grouped heads, D = 256), q float32 [t, hq x d] (rows `qs` floats apart), k and v
 * float32 [t, hkv x d] (rows `kvs` apart), out float32 [t, hq x d]: query blocks of 512 rows against the keys up to
 * their end as half products on oneDNN, a masked float32 softmax between them. The shared causal attention walks the
 * keys one at a time a (head, query) and takes 0.3 s a layer at 4,000 tokens. _supported: t from 1 to 4096, d = 256,
 * hq a multiple of hkv. */
int ns_image_moss_silo_attn_supported(ns_gpu* g, int64_t t, int64_t hq, int64_t hkv, int64_t d);
int ns_image_moss_silo_attn_causal(ns_gpu* g, const float* q, const float* k, const float* v, int64_t t, int64_t hq, int64_t hkv, int64_t d, int64_t qs, int64_t kvs,
                                   float* out);

#ifdef __cplusplus
}
#endif
#endif
