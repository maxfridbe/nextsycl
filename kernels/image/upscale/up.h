/* up.h: the upscalers' own kernels (kernels/image/upscale): Real-ESRGAN's and waifu2x's ncnn graphs run on the GPU -
 * every activation channels-last IEEE half [H, W, C], float32 arithmetic - beside the shared diffusion kernels
 * (nsd.h: the convolutions are its strided im2col + GEMM). Bound by name from image/upscale/src/ffi.rs. Every call is
 * queued on the GPU's queue (the caller syncs) and returns 0, or -1 with the reason in ns_last_error(). */
#ifndef NS_IMAGE_UP_H
#define NS_IMAGE_UP_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* A tile of the picture: out [th, tw, 3] half = src[y0 + y, x0 + x, c] / 255 (src u8 [H, W, ch], ch 3 or 4, its alpha
 * not read). Rows and columns outside the picture: border 0 mirrors them (abs(i), then (n-1) - abs(i - (n-1)):
 * Real-ESRGAN's preprocess), 1 clamps them to the edge (waifu2x's) */
int ns_image_up_pre(ns_gpu* g, const uint8_t* src, int64_t H, int64_t W, int64_t ch, int64_t x0, int64_t y0, int64_t tw, int64_t th,
                    int border, void* out);
/* The tile's alpha as half [th, tw] (the picture's channel 3 / 255, rows and columns clamped) */
int ns_image_up_pre_alpha(ns_gpu* g, const uint8_t* src, int64_t H, int64_t W, int64_t x0, int64_t y0, int64_t tw, int64_t th, void* out);
/* Into the picture: dst u8 [DH, DW, dch] at (dx, dy) for w x h pixels = floor(src[sy + y, sx + x, c] * 255 + 0.5) clamped to
 * 0..255, src half [SH, SW, 3]; alpha (half [h, w] or NULL: then dch 3) becomes channel 3 the same way */
int ns_image_up_post(ns_gpu* g, const void* src, int64_t SW, int64_t sx, int64_t sy, const void* alpha, uint8_t* dst, int64_t DW,
                     int64_t dch, int64_t dx, int64_t dy, int64_t w, int64_t h);
/* out [H + t + b, W + l + r, C] half: x [H, W, C] half with zero borders */
int ns_image_up_pad(ns_gpu* g, const void* x, int64_t H, int64_t W, int64_t C, int64_t t, int64_t l, int64_t b, int64_t r, void* out);
/* out [(H - 1) s + 1, (W - 1) s + 1, C] half: x's pixels s apart, zeros between (a transposed convolution's input) */
int ns_image_up_dilate(ns_gpu* g, const void* x, int64_t H, int64_t W, int64_t C, int64_t s, void* out);
/* x [N, C] half, in place: kind 1 relu, 2 leaky relu (slope a), 3 prelu (slopes [C] float32, or one when n_slopes 1),
 * 4 sigmoid */
int ns_image_up_act(ns_gpu* g, void* x, int64_t N, int64_t C, int kind, float a, const float* slopes, int64_t n_slopes);
/* out [H r, W r, C] half = pixel_shuffle(x [H, W, C r r]) (PyTorch's channel order: c r r + dy r + dx) */
int ns_image_up_pixel_shuffle(ns_gpu* g, const void* x, int64_t H, int64_t W, int64_t C, int64_t r, void* out);
/* out [Ho, Wo, C] half = x [H, W, C] resized: kind 1 nearest, 2 bilinear (half-pixel centres, ncnn's align_corner 0),
 * 3 bicubic (a -0.75, ncnn's) */
int ns_image_up_resize(ns_gpu* g, const void* x, int64_t H, int64_t W, int64_t C, int64_t Ho, int64_t Wo, int kind, void* out);
/* out = ca a + cb b (op 0), a b (op 1), max(a, b) (op 2): n halves (Eltwise, BinaryOp) */
int ns_image_up_add(ns_gpu* g, const void* a, const void* b, int64_t n, int op, float ca, float cb, void* out);
/* dst [P, Cd] half: channels [off, off + Cs) = src [P, Cs] (a Concat's part) */
int ns_image_up_copy_channels(ns_gpu* g, const void* src, int64_t P, int64_t Cs, void* dst, int64_t Cd, int64_t off);
/* out [h, w, C] half = x [H, W, C] from (x0, y0) */
int ns_image_up_crop(ns_gpu* g, const void* x, int64_t W, int64_t C, int64_t x0, int64_t y0, int64_t w, int64_t h, void* out);
/* out [C] float32 = the mean of x [P, C] half over its P pixels */
int ns_image_up_mean(ns_gpu* g, const void* x, int64_t P, int64_t C, float* out);
/* out [O] float32 = act(b + w x): w float32 [O, I], x float32 [I], b float32 [O] or NULL; kind as ns_image_up_act */
int ns_image_up_dense(ns_gpu* g, const float* x, int64_t I, const float* w, const float* b, int64_t O, int kind, float a, float* out);
/* x [P, C] half, in place: x * s[c] (s float32 [C]) */
int ns_image_up_scale(ns_gpu* g, void* x, int64_t P, int64_t C, const float* s);

#ifdef __cplusplus
}
#endif
#endif
