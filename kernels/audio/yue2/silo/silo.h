/* silo.h: YuE2's silo (dist/silo/libnextsycl-yue2.so): its Q8_0 matrices repacked at load, the quants apart from the
 * scales (lossless). A Q8_0 block's 34 bytes leave its quants 2-aligned: the engine's own kernels read them two bytes
 * at a time; here a row's quants are read as aligned 16-byte vectors. Opened by the engine (audio/yue2/src/ffi.rs); without
 * it (or NS_SILO=0) the engine keeps the matrices as stored. 0 or -1 (ns_last_error). */
#ifndef NS_AUDIO_YUE2_SILO_H
#define NS_AUDIO_YUE2_SILO_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* 1 when this GPU runs these kernels for rows of K values: sub-groups of 16, K a multiple of 32 */
int ns_audio_yue2_silo_supported(ns_gpu* g, int64_t K);

/* A repacked matrix [N, K]: quants [N][K] int8 from its start, scales [N][K / 32] half from ns_audio_yue2_silo_scales_at */
size_t ns_audio_yue2_silo_scales_at(int64_t N, int64_t K);
size_t ns_audio_yue2_silo_bytes(int64_t N, int64_t K);
int ns_audio_yue2_silo_pack(ns_gpu* g, const void* q8_0, int64_t N, int64_t K, void* dst);

/* out[m, n] (+)= sum_k x[m, k] w[n, k], as ns_audio_yue2_gemv_q8: m < M <= 8, x float32 [M, K], out float32 rows ldo apart;
 * wq, ws the rows' quants and scales (a repacked matrix's, or past its first rows) */
int ns_audio_yue2_silo_gemv(ns_gpu* g, const float* x, int64_t M, int64_t K, const void* wq, const void* ws, int64_t N, float* out, int64_t ldo, int acc);

/* The whole matrix as half [N, K] (a prompt or flow pass's GEMM) */
int ns_audio_yue2_silo_half(ns_gpu* g, const void* wq, const void* ws, int64_t N, int64_t K, void* out);

#ifdef __cplusplus
}
#endif
#endif
