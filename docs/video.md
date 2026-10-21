# Video

[← nextsycl](../README.md)

MiniMax H3's engine, daemon, studio and tools (the sycl-h3 studio) as `nextsycl video`:

```sh
nextsycl models pull minimax-h3 [--dir DIR] [--from DIR]      # --from: adopt copies already on the machine
nextsycl video start [--model ID] [--engine ID]... [--gpu N ...] [--shared-gpu N ...]   # the daemon, in its container
nextsycl video serve [--bind ADDR] [--port N]                   # the studio: H3's web front end and clip queue
nextsycl video job generate --prompt "..." --width 768 --height 576 --seconds 5 --steps 8 --out DIR/clip.mp4 [-f]
nextsycl video ps [-a] | inspect ID | cancel ID... | rm ID... | status | gpus | unload [--gpu N] | logs [--web]
nextsycl video speech FILE --character NAME | scene SCENE.json | join PREFIX | speechpct CLIP... | plan measure
nextsycl video job <kind> ... --here                            # a job in this process (load, run, exit)
nextsycl video stop [--web | --all]
```

- **The engine** (`video/h3`, the `VideoEngine` contract): H3's h3-core, its jobs (generate, encode, decode,
  denoise, check-block, bench-blocks) and mp4 writer, on the diffusion kernels of `libnextsycl-video.so` (H3's under
  the `nsd_` prefix; SageAttention from `libnextsycl-flash.so`). Its options: `nextsycl video engines`.
- **The daemon** (`glue/serve` video::daemon): H3's queue, one worker process a GPU slot (it loads on the first
  job, unloads after `NS_VIDEO_IDLE`), the card lock and the front end's model switch for a GPU shared with chat.
- **The studio** (`video serve`, a host process): H3's front end (`wfe/video`, TSX on snabbdom) and its legacy API,
  the clip queue, projects, scenes, films, the language-model switch (`NS_VIDEO_LLM_MODES`).
- **Samplers**: a job's `sampler` and `schedule` take ComfyUI's names (`--sampler euler_ancestral`, `dpmpp_2m`,
  `uni_pc` ...; the same 29 samplers and 10 schedules as the image engine); without them, Euler on H3's own shifted
  schedule. The studio's Create form has both. A 3 s 768x576 clip, 8 steps on the B70 takes 46-52 s with euler,
  euler_ancestral or dpmpp_2m alike (one model call a step; heun-type samplers take two or three).
- Settings: `NS_VIDEO_MODEL`, `NS_VIDEO_ENGINES`, `NS_VIDEO_OUT`, `NS_VIDEO_GPUS`, `NS_VIDEO_SHARED_GPUS`,
  `NS_VIDEO_IDLE`, `NS_VIDEO_GPU_LOCK`, `NS_VIDEO_LLM_SWITCHER`, `NS_VIDEO_MODELS_DIR` (seen as /models: H3's paths
  in scene files, the pixel upscalers), `NS_VIDEO_LISTEN` / `_PORT`, `NS_VIDEO_STUDIO_DIR`, `NS_VIDEO_LLM_MODES`,
  `NS_VIDEO_TEMPLATES`, `NS_VIDEO_CHARACTERS` (`docs/video/characters.example.json`).
- Checked against H3's own `h3d` on the B70, the same clip: the latents differ from H3's by as much as two H3 runs
  differ from each other (cosine 0.996 - H3 is not bit-reproducible run to run), the frames the same scene; 8.0 s a
  denoising step either way at 896x672, 4.5 s (19,191 tokens).

What a clip can be made from (ComfyUI's H3 nodes, each checked against them where noted):

| Mode | How | Keys (job spec; the studio's form has each) |
|---|---|---|
| text to video+audio | the prompt | `prompt` |
| image to video, first / last frame, both | keyframes pinned at frame 0 / the last frame; their pictures shown to the text encoder too (`<Picture i>`, as ComfyUI's image-to-video node) | `first_frame`, `last_frame`, `te_pictures` |
| motion guide, clip continuation | a clip's last frames (22) anchored as one moving keyframe, its sound tail - how a film carries on; the last latent of the clip before (jumps in exposure: chain a PNG) | `guide_clip`, `first_audio`, `first_latent` |
| references (Ref2VA) | up to 9 pictures, 3 clips (their sound too) and 3 sounds named `<Picture i>`, `<Video k>`, `<Audio j>` in the prompt; the Ref2VA denoiser (`minimax-h3-ref2va`). The 32B text encoder's vision tower (`visual.*`, range-fetched from the bf16 file) runs from crates/qwen3vl - checked against ComfyUI: cosine 0.99994 | `ref_images`, `ref_videos`, `ref_audio`, `ref_image_size` |
| effects | Comfy-Org's ten embeddings: `embedding:minimaxh3_bullet_time` in the prompt (ComfyUI's splitting; cosine 0.999999) | `embeddings` (from the model's roles) |
| ControlNet (Fun ControlNet-Union 2.0) | a control video - canny, depth, HED, MLSD or pose frames, made beforehand - the clip follows | `control_video`, `control_strength`, `control_start`, `control_end` |
| video inpainting | a mask (white: regenerate; a picture or a video) over a source video, through the same ControlNet | `control_mask`, `control_source` |
| regenerate a stretch / box, extend | a clip's latents kept outside the seconds and box asked for | `source`, `regen`, `regen_box` |

![the video studio](screenshots/video-studio.webp)
