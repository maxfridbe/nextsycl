/* ns.h: libnextsycl's C ABI - what the Rust runtime (crates/ns-sys) calls.
 *
 * Every call returns 0 on success, or -1 with the reason in ns_last_error() (per thread). Device memory is USM
 * device memory of one GPU's own context: each GPU is opened with a context of its own, never the platform's
 * default context, which spans every GPU and makes xe mirror each card's allocations into host RAM (measured
 * 2026-10-05: ~45 GB of host RAM for 40 GB of model on two cards). Copies between GPUs therefore go through
 * host memory (ns_copy_peer).
 */
#ifndef NS_H
#define NS_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

const char* ns_last_error(void);
const char* ns_version(void);

/* the Level Zero GPUs, in the order ns_gpu_open takes; name of index i into buf */
int ns_gpu_count(void);
int ns_gpu_name(int index, char* buf, size_t len);
/* opens GPU `index` with its own context and one in-order queue */
int ns_gpu_open(int index, ns_gpu** out);
void ns_gpu_close(ns_gpu* g);
/* the queue, as the imported kernels take it (`void* stream` = sycl::queue*) */
void* ns_gpu_queue(ns_gpu* g);
/* bytes: total device memory, and free (-1 when the driver does not say) */
int ns_gpu_memory(ns_gpu* g, uint64_t* total, int64_t* free_bytes);

int ns_alloc(ns_gpu* g, size_t bytes, void** out);
int ns_free(ns_gpu* g, void* p);
/* pinned host memory of this GPU's context: the fast source and target of its copies */
int ns_alloc_host(ns_gpu* g, size_t bytes, void** out);
int ns_free_host(ns_gpu* g, void* p);
/* queued on the GPU's queue (in order); ns_sync waits for everything queued */
int ns_copy_to(ns_gpu* g, void* dst, const void* src_host, size_t bytes);
int ns_copy_from(ns_gpu* g, void* dst_host, const void* src, size_t bytes);
int ns_copy_dev(ns_gpu* g, void* dst, const void* src, size_t bytes);
int ns_fill(ns_gpu* g, void* dst, uint8_t value, size_t bytes);
int ns_sync(ns_gpu* g);
/* device memory of one GPU to another's, through `staging`: host memory of `bytes` (pinned memory belongs to one
 * context, so it is pinned for at most one of the two copies; plain memory works for both, slower); waits */
int ns_copy_peer(ns_gpu* to, void* dst, ns_gpu* from, const void* src, size_t bytes, void* staging);

#ifdef __cplusplus
}
#endif
#endif
