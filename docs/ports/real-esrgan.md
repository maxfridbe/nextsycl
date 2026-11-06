## Real-ESRGAN (realesrgan-ncnn-vulkan's graphs): the speed pass

image/upscale runs the ncnn graphs and half weights that realesrgan-ncnn-vulkan ships (x4plus: RRDB, 23 blocks;
x4plus-anime: 6 blocks; realesr-animevideov3: SRVGG compact at 2x / 3x / 4x) in tiles of 200 pixels with 10 of
mirrored context, as the program does. Until this pass its convolutions were the shared ones (nsd: a padded copy of
the input, strided im2col + GEMM, then an activation pass), a wait for the card after every tile, and Concat copying
every channel of a dense block again at each step.

### Checked against the program

`nextsycl image upscale-check PICTURE REFERENCE.png --model real-esrgan --variant V --scale S`: PSNR and the worst
pixel against realesrgan-ncnn-vulkan's own output (the release's program, from the cached release zip, on hardac1;
its sample pictures turned into PNG first so both sides decode the same pixels: input.png 220 x 220, input2.png
256 x 256, a frame of its onepiece_demo.mp4 640 x 480). The program on the AMD iGPU (RADV) wrote x4plus black, so that
reference is the program on the B70 (Vulkan, -g 1); the others are the iGPU's.

| variant | picture | main (shared convolutions) | silo (step 1) | steps 3-5 (the final build) | whole frame (rejected) |
|---|---|---|---|---|---|
| x4plus | input 220 -> 880 | 56.89 dB, worst 6/255 | 56.90 dB, worst 6 | 56.97, 6 | 48.59 dB, worst 41 |
| x4plus-anime | input2 256 -> 1024 | 58.78 dB, worst 8 | 58.78, 8 | 58.79, 9 | 55.43, 49 |
| animevideov3 x2 | onepiece 640x480 | 60.46 dB, worst 4 | 60.47, 4 | 60.47, 4 | 60.05, 8 |
| animevideov3 x3 | | 60.24, 4 | 60.25, 4 | 60.25, 4 | 59.88, 8 |
| animevideov3 x4 | | 60.98, 4 | 60.99, 4 | 60.99, 4 | 60.51, 9 |

The same on both cards (the B65 and the B70 give the same numbers to the digit). The residual in the convolution
(step 4) sums in float32 before rounding to half: a hair closer to the program (56.90 -> 56.97 dB).

### Speed log (frames/s through `video.upscale`, the image server's own loop: ffmpeg decode, the engine a frame at a time, x264)

Clips: h3_20261105_134939 (864 x 480, 243 frames) and h3_20260919_014942 (1344 x 768, 243 frames); x4plus and
x4plus-anime on their first 48 (864) / 24 (1344) frames. B70 (gpu 0) unless said.

| # | change | av3 x2 864 | av3 x3 | av3 x4 | x4plus 864 | x4plus-anime 864 | av3 x2 1344 | x4plus 1344 |
|---|---|---|---|---|---|---|---|---|
| 0 | main: the shared convolutions (padded copy, im2col + GEMM, activation pass), a wait after each tile | 32.9 | 28.7 | 25.9 | 0.92 | 2.66 | 14.3 | 0.38 |
| 1 | silo (`kernels/image/upscale/silo/conv.cpp`): oneDNN's convolutions (implicit GEMM) with the zero padding in the primitive and the activation as a post-op (LeakyReLU, ReLU, sigmoid; SRVGG's PReLU fused into the convolution before it) | 74.5 | 53.7 | 43.2 | 1.83 | 5.03 | 33.3 | 0.75 |
| 2 | no wait for the card after each tile (the queue is in order; `NS_UP_SYNC=1` brings it back) | 74.1 | 53.6 | 44.0 | 1.84 | 5.08 | 33.4 | 0.75 |
| 3 | the dense blocks' concatenations in place (below): measured on the B65 (the B70 kept for the front door from here) | | | | | | | |
| - | the whole frame as one tile (`NS_UP_TILE=0`): faster for animevideov3 (86.4 / 60.6 / 47.3, 1344: 41.6) but further from the program (above: x4plus 48.6 dB, its tiles' context is part of the picture it makes) and slower for x4plus while its concatenations were copies; not the default | | | | | | | |

### The B65 (gpu 1: the measurements after 15:30 on 2026-11-06, the B70 kept for the front door)

The B65 has 98.3 TFLOPS of half (fewer Xe cores than the B70's 183.5) and the same memory: its figures are its own.
"main" is this branch's build with main's paths switched back (`NS_SILO=0`: the shared convolutions and every
concatenation copied; `NS_UP_SYNC=1`: a wait after every tile) - the code main runs, built once.

| # | change | av3 x2 864 | av3 x3 | av3 x4 | x4plus 864 | x4plus-anime 864 | av3 x2 1344 | x4plus 1344 |
|---|---|---|---|---|---|---|---|---|
| 0 | main | 20.1 | 18.2 | 17.4 | 0.55 | 1.63 | 8.7 | 0.23 |
| 1+2+3 | the silo's convolutions (step 1), no wait a tile (2), the dense blocks' concatenations in place (3) | 51.1 | 39.9 | 36.0 | 2.06 (without 3: 1.20) | 4.99 | 23.2 | 0.88 |
| 4 | RRDB's residual (0.2 x block + x) summed in the block's last convolution (oneDNN: linear, binary add post-ops) | | | | 2.18 | 5.14 | | 0.93 |
| 5 | the silo's copy (a block's input into its buffer) and nearest upsampling in 16-byte pieces, no divisions | 53.8 | 41.2 | 36.8 | 2.53 | 6.53 | 24.6 | 1.09 |
| | B65, main -> step 5 | 2.7x | 2.3x | 2.1x | 4.6x | 4.0x | 2.8x | 4.7x |

Busy (the B65's compute engine, fdinfo): 88 % at step 5 for animevideov3 and x4plus; one CPU core (ffmpeg's
decode and encode, the frames' bytes, on the server's thread).

### Where the time goes (`NS_UP_PROFILE=N`: each layer timed alone, B70)

animevideov3 x2, 48 frames of 864 x 480 (15 tiles a frame):

| step | main: s (TFLOPS, % of 183.5) | silo: s (TFLOPS, %) |
|---|---|---|
| conv 3x3 64 -> 64 + PReLU (16 a tile) | 1.157 (25.5, 13.9 %) | 0.437 (67.4, 36.7 %) |
| conv 3x3 3 -> 64 + PReLU | 0.123 (0.7) | 0.097 (0.9) |
| conv 3x3 64 -> 48 (to the pixel shuffle) | 0.129 (10.7) | 0.095 (14.6) |
| interp (the nearest-upsampled input, added) | 0.072 | 0.072 |
| pixel shuffle | 0.045 | 0.045 |

x4plus, 12 frames of 864 x 480:

| step | main: s (TFLOPS, %) | silo: s (TFLOPS, %) |
|---|---|---|
| concat (the dense blocks: each step copies all the channels so far) | 3.157 (315 GB/s) | 3.177 (313 GB/s): 42.5 % of the time |
| conv 3x3 192 -> 64 | 2.773 (34.4, 18.7 %) | 1.075 (88.7, 48.3 %) |
| conv 3x3 160 / 128 / 96 / 64 -> 32 + LeakyReLU | 6.02 (17-19) | 1.875 (45-70) |
| eltwise (the blocks' 0.2 x residual) | 0.502 | 0.519 |

x4plus on the B65 after step 5, 12 frames of 864 x 480 (the concatenations gone, the residuals inside the
convolutions): conv 192 -> 64 + residual 1.79 s (53 TFLOPS, 54 % of the B65's 98.3), the 32-channel convolutions
0.64-0.77 s each kind (25-53 %), 64 -> 64 0.37 s, the outer residuals (one an RRDB) 0.17 s; animevideov3 x2: the 64 ->
64 + PReLU convolutions 61 % (48 TFLOPS, 49 %), 64 -> 48 12 %, 3 -> 64 10 % (0.8 TFLOPS: three input channels), the
pixel shuffle 8 %, the input's nearest upsampling 7.5 %.

### Open

- The tiles: 200 pixels with 10 of context (the program's) - 15 tiles a 864 x 480 frame, each convolution a small
  problem (220 x 220). Running a frame's same-sized tiles as one batch (oneDNN's N) would give the same output with
  bigger kernels; the whole frame as one tile is faster for animevideov3 but changes the output (above).
- Frames one at a time through the image server's loop (decode, upscale, encode on one thread): the host's share is
  small at these speeds (the card 88 % busy) but a second frame in flight would hide it.
- The 3 -> 64 first convolution (0.8 TFLOPS) and the pixel shuffle: a direct kernel each.
- Two cards for a clip (frames alternating) - not started.
