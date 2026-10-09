/* example.h: the audio template engine's kernels (audio/example) - its own C ABI, bound by its crate by name
 * (audio/example/src/ffi.rs). An engine's symbols are named ns_<kind>_<arch>_*, so engines never collide. */
#ifndef NS_AUDIO_EXAMPLE_H
#define NS_AUDIO_EXAMPLE_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* x[i] *= s for i < n, x device memory of g; queued on g's queue (the caller syncs). 0, or -1 and ns_last_error() */
int ns_audio_example_scale(ns_gpu* g, float* x, int64_t n, float s);

#ifdef __cplusplus
}
#endif
#endif
