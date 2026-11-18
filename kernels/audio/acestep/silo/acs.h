/* acs.h: ACE-Step 1.5's silo (dist/silo/libnextsycl-acestep.so): the Oobleck VAE's 1-D convolutions on oneDNN.
 * The shared nsd_conv1d / nsd_conv_transpose1d are a plain loop an output sample (a 262 s song: ~3 min to decode,
 * ~2 to encode, 2% of the card's float32 rate); here each is one oneDNN primitive, float32 in and out with float32
 * accumulation (fpmath strict: the same products, summed in another order). Primitives are made once a shape and kept
 * (bounded), each weight reordered once to the primitive's layout (keyed by its pointer: ns_acs_release forgets them
 * when the engine's weights go), all of a card's primitives sharing one user scratchpad. 0 or -1 (ns_last_error). */
#pragma once
#include "ns.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 1 when this silo runs on the card (a oneDNN engine on its device and queue); 0 sends the VAE to the shared kernels */
int ns_acs_supported(ns_gpu* g);

/* x [Ci, L] float32 (one signal, channels first) through a convolution: transposed 0 - w [Co, Ci, K] (the checkpoint's
 * layout), out[o, n] = bias[o] + sum_i,k w[o, i, k] x[i, n * stride - pad + k * dil] (zeros outside), Lo outputs;
 * transposed 1 - w [Ci, Co, K], out[o, m] = bias[o] + sum over m = n * stride - pad + k of x[i, n] w[i, o, k]
 * (dil must be 1). bias may be NULL. Queued on the card's queue (the caller syncs). */
int ns_acs_conv1d(ns_gpu* g, int transposed, const float* x, int64_t Ci, int64_t L, const float* w, int64_t Co, int64_t K,
                  const float* bias, int64_t stride, int64_t dil, int64_t pad, float* out, int64_t Lo);

/* The transposed convolution above (dil 1), made as the `stride` phases of its output: each phase r (outputs m with
 * (m + pad) % stride == r) is a plain convolution of x with the taps k = r + stride j, J = ceil(K / stride) of them -
 * all phases one oneDNN convolution (Ci -> stride x Co channels, J taps; oneDNN's own transposed one runs at a
 * fraction of its rate on these shapes) into tmp, then interleaved into out. The same products and sums. tmp: at least
 * stride x Co x (L + J - 1) floats (the card's; free during the call); smaller, the transposed primitive instead. */
int ns_acs_conv_transpose1d(ns_gpu* g, const float* x, int64_t Ci, int64_t L, const float* w, int64_t Co, int64_t K, const float* bias,
                            int64_t stride, int64_t pad, float* tmp, int64_t tmp_floats, float* out, int64_t Lo);

/* What the silo holds on the card, in bytes: the weights in the primitives' layouts and the phases' (with their bias),
 * and the scratchpad */
int ns_acs_held(ns_gpu* g, int64_t* weights, int64_t* scratch);

/* The scratchpad given back (after the queue's work): the end of a pass; the next call makes it again */
int ns_acs_trim(ns_gpu* g);

/* Everything kept for the card (the primitives, the reordered and phase-split weights, the scratchpad) freed, after the queue's work:
 * a VAE's weights are going (a later one could be given the same addresses) */
int ns_acs_release(ns_gpu* g);

#ifdef __cplusplus
}
#endif
