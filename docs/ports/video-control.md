## video.control: control videos for the H3 Fun ControlNet (canny, gray, Depth Anything V2 Small)

Goal: level with nextcuda's `video.control` (a control video made from a source clip) and `video.generate`'s
`control_video` taking it, with nextcuda's shapes exactly (its glue/serve/src/home/spec.rs, methods.rs, control.rs and
video/studio/control.rs at a9e3e43, 47850b2, f0fc4b9, aff685a, b315f61, read from github.com/maxfridbe/nextcuda).

### What was already here

The H3 Fun ControlNet-Union 2.0 and the inpaint mode (crates/h3 `control.rs`, `jobs.rs` `clip_control_hint`), the job
keys and `video.generate`'s `control_video`, `control_mask`, `control_source`, `control_strength`, `control_start`,
`control_end` (ported 2026-10-09 with every ComfyUI H3 mode), the `controlnet` role on the INT8 / Q6_K / Q4_K_M cards.

### What was added

- **`video.control`** (methods.rs `VideoControlReq`, spec.rs answer and errors - nextcuda's): `file` (a clip in the
  clips directory), `control` (canny, gray, depth; pose refused 400), `model` (depth), `width` / `height` (256-1344,
  multiples of 32; none: the source's aspect at about 768 x 576), `frames` (17k + 5, 22-362; none: the most the source
  has from `start_s`), `start_s`, `low` / `high` / `normalize` (canny); the job fields for depth (`wait`, `label`,
  `reentrant_id` ...) and nextsycl's `rsvp_s` / `max_willing_to_wait_s`. The answer: `name`
  (`ref_<unix>_<stem>-<control>.mp4`, what `control_video` takes), `file` (its URL), `control`, `width`, `height`,
  `frames`, `fps`, `seconds`, `source_frames`, `source_seconds`, `start_s`, `low`, `high`, `took_s`, `levels` (canny),
  and for depth `model`, `range`, `frame_seconds`, `job`.
- **canny and gray** on the CPU in the studio (glue/serve/src/video/studio/control.rs, nextcuda's file: OpenCV's Canny
  with its L2 gradient and hysteresis, BT.601 luma; the clip read at 24 fps from `start_s`, scaled to cover and cut in
  the middle, a short source's last frame held). ffmpeg through `video::media::tool`: hardac1's ffmpeg-free cannot
  decode H.264, so the host side runs the video daemon's container's.
- **depth**: an image-queue job (`jobs::runs_as_job`, `kind_of_method` image) like video.upscale. The front door
  (home/control.rs) picks the depth model (capability `depth`, 409 when none is enabled), checks the source and
  settings at submit, and sends the image server POST /v1/videos/depth {path, name, width, height, frames, start_s}; the
  server reads every frame, the engine estimates its depth, `make_depth` keeps them in 16 bits and writes the clip
  scaled by one range (0.5th-99.5th percentile of all frames' values, near bright). The image container sees the clips
  read-only: it writes the video into its own output directory and the front door moves it beside the clips.
- Features `video.control`, `video.control.canny`, `video.control.gray` always; `video.control.depth` only while a
  depth model is enabled (nextcuda b315f61).
- **Depth Anything V2 Small** (image/depthanything, card `depth-anything-v2-small`, Apache-2.0; docs/architecture.md
  has the engine). Its attention is float32 products in its silo (see there for the speed story).

### Checks

- **Parity** (`nextsycl image depth --check`, reference/depthanything/ref.py: transformers 5.19 on the CPU, float32,
  run with `python -I` from its own directory on hardac1): six frames of two H3 clips - three 2560 x 1440 frames
  (924 x 518 into the network), two 864 x 480 (938 x 518), one 896 x 512 control canvas (910 x 518), on the B65. The
  input identical; embeddings 4.8e-8 relative; the four taps 2.7e-6; the last fusion 1.6e-6; the head's output 1.5e-6
  (largest 1.96e-4 absolute); the depth at the picture's size 1.7e-6 relative, at most 2.2e-4 absolute on maps of
  0-5.4 to 0.36-7.6 (about 4e-5 of the range); the 8-bit map (own min-max) equal on 99.993 % of pixels or more, at
  most 1 apart. (The shared kernel's attention, tried: 73 ms a pass but 1.4e-4 relative, 9.7e-3 absolute - kept out.)
- **Speed** (B65, `NS_PROFILE=1`): 1.44 s a picture as first written (the per-query attention 46 ms a layer, and
  slice copies on the card of 160 ms each); the taps now views, the attention as products: ~0.10 s (backbone ~58 ms,
  neck and head ~42 ms).
- **The clip path**: the image server's POST /v1/videos/depth (the build copy's server, B65) on an 864 x 480 H3 clip from
  1 s, 768 x 448 x 73 frames: 7.5 s (0.099 s a frame), range 0.303-6.803 (the same as with the first, slower kernels).
- **End to end** (B65, the build copy, `video job generate --here`, minimax-h3 INT8, 768 x 448, 2 s, seed 1234, "a stone
  temple courtyard at dusk ..."): with that depth video as `control_video` (strength 1; the ten control blocks loaded in
  5.1 s) the clip took 91 s, without it 60 s. Each clip's own depth (the same route) against the control's, Pearson
  over 56 frames at 192 x 112: **0.870** with the control (frames 0.85-0.88), **0.235** without - the clip follows the
  depth.
- Not run on hardac1: canny and gray through the front door and the studio (their ffmpeg is the video daemon's
  container's there); the unit tests run them through a local ffmpeg.
- **Tests**: glue/serve/tests/front_door.rs `video_control_depth_needs_a_depth_model_and_is_an_image_job` (409 without
  a depth model, depth absent from the features, 400 for a model that does not estimate depth, the source and settings
  checked at submit, 404 for a missing clip; with one: the feature listed and a job of the image queue; the model's
  capabilities and methods), home/control.rs unit tests, studio control.rs's (canny, gray, the depth clip through
  ffmpeg with a stand-in estimator, a cancel).
