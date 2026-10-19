/* vcs.h: VoxCPM2's silo (dist/silo/libnextsycl-voxcpm2.so): a patch's flow-matching sampler on the GPU - the DiT's
 * evaluations and the guided Euler update between them (CFG-zero*: the unguided velocity scaled to fit the guided
 * one), recorded once a request as a SYCL command graph and replayed a patch at a time, so the host waits once a
 * patch instead of once an evaluation. 0 or -1 (ns_last_error). */
#pragma once
#include "ns.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 1 when this silo takes a patch of these sizes: `n` values (latents x features) at most 4,096 */
int ns_vcs_supported(int64_t n);

/* One guided Euler step of the sampler, as the host made it (the same float32 operations in the same order: the
 * same bits): v [2, n] - the guided velocity, then the unguided one, both in x's layout; with st = (pos . neg) /
 * (|neg|^2 + 1e-8), summed in the order `order` gives (order[j]: the index of the j-th value summed; the host's
 * channels-first order), x -= dt * (neg * st + cfg * (pos - neg * st)) */
int ns_vcs_cfg_euler(ns_gpu* g, float* x, const float* v, const int32_t* order, int64_t n, float dt, float cfg);

/* A SYCL command graph of the queue's work between begin and end (recorded, not run), replayed by launch */
int ns_vcs_graph_begin(ns_gpu* g, void** graph);
int ns_vcs_graph_end(ns_gpu* g, void* graph, void** exec);
int ns_vcs_graph_launch(ns_gpu* g, void* exec);
void ns_vcs_graph_free(void* exec);

#ifdef __cplusplus
}
#endif
