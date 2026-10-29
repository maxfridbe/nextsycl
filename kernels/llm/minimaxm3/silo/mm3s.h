/* mm3s.h: the minimaxm3 silo (dist/silo/libnextsycl-minimaxm3.so, opened at load; without it, or with NS_SILO=0, the engine runs
 * the shared kernels of ../mm3.h). Kernels tuned for this model's shapes alone. */
#ifndef NS_MM3S_H
#define NS_MM3S_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* Whether ns_mm3s_attend takes this geometry: heads of 128 values, 16 query heads a key/value group */
int ns_mm3s_attend_supported(int64_t H, int64_t G, int64_t head);

/* ns_llm_minimaxm3_attend (mm3.h) for a prompt pass: the same arguments and result (the scores' sums in another order), a
 * work-group a (query row, key/value group) sharing each staged tile of 64 keys and values among the group's 16 heads */
int ns_mm3s_attend(ns_gpu* g, const float* q, const uint16_t* kc, const uint16_t* vc, const int32_t* sel, int64_t K, float* out, int64_t T, int64_t H, int64_t G,
                   int64_t pos0, float scale);

/* The same for a few query rows (decode): the keys split in chunks of 128, a work-group each (S = ceil(keys / 128) chunks,
 * keys = K * 128 selected or pos0 + T), their partial softmax states in `part` (T * G * S * 16 * 130 floats), merged by a second kernel */
int ns_mm3s_attend_split(ns_gpu* g, const float* q, const uint16_t* kc, const uint16_t* vc, const int32_t* sel, int64_t K, float* out, int64_t T, int64_t H, int64_t G,
                         int64_t pos0, float scale, int64_t S, float* part);

#ifdef __cplusplus
}
#endif
#endif
