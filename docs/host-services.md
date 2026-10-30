# Host services

[← nextsycl](../README.md)

Three small services sit beside the model servers, all in the program (no Python on the serving path):

```sh
nextsycl serve [--port 8000] [--host 0.0.0.0] [--serve-host 0.0.0.0] [--idle-minutes 10]
nextsycl switch [--port 8001] [--upstream 127.0.0.1:8085] [--studio http://127.0.0.1:8090/rpc/llm.mode] [--alias OLD=NEW]...
                [--images 127.0.0.1:8086]
nextsycl gpustat [--out /run/gpustat.json] [--interval 3] [--pci ADDR] [--once]     # as root
```

- **`serve`** (:8000, `docs/host/nextsycl-serve.service.example`): the front door.
  - **The page**: every GPU's memory by process, and below it, hatched, what the enabled models are vouched for (as
    seen while they ran, else an estimate from their files; chat models and video engines count once a card - one
    runs at a time - and an overbooked card is flagged: whichever loads first holds it). Every registered model by
    kind: enable it on its GPU(s), start it now (the button spins until it answers), stop it, disable it, its page.
  - **Model pages** through the door: an image, audio or 3D model's own page is `/open/<id>/` (the model starts if it
    is not up; every request under it, the page's API calls too, is passed to its server as `/<rest>`). Those servers
    check no token, so run the door with `--serve-host 127.0.0.1` (the example unit does): they listen on this
    machine only and another machine reaches them here. The video studio and the chat page are sent to by redirect.
  - **Enabled** models load on their first request - an API call, their page, Open WebUI through the switcher - and
    unload after `NS_IDLE_MINUTES` (10) without one (the image and music servers exit, `--idle-exit`; the switcher
    sets the chat mode to none; the video daemon's workers let their card go), staying enabled. A load waits for its
    card to have the model's memory free plus 1.5 GiB (past the card the xe driver spills into host memory); idle
    work holding it is stopped for it at once (picture and sound servers without a request, the video worker while
    the studio has nothing to make, a chat model answering nothing that no `X-Keep-Up-S` keeps - never a render or a
    request), and it answers
    503 `no_room` when the room does not come free (a job waits up to 30 minutes first, saying what for; a chat
    load, like the others, 90 s). A model whose card says `ram_gib` also waits for host memory by the same rule:
    MemAvailable, less the warm cache of a running model that says `ram_cache_gib` (the kernel counts that cache as
    available, and the model will fill it again), must leave `ram_gib` plus 4 GiB. Idle holders go for it, kept or
    working ones stay; a model without `ram_gib` skips the check. A video job waiting on any card, shared or not,
    gets the room rule too: idle chat on that card goes (however it was started), a kept or answering one stays and
    `video.status` says `waiting_for` until the 30-minute limit fails the job with `no_room`. The shared GPU also
    keeps its lock and its `llm.mode none` hook, which do not look at `X-Keep-Up-S`. **Disabled**: unloaded, gone
    from Open WebUI's list and from the API.
    The registry's `enabled` and `gpus` are the state (`nextsycl models enable|disable` agree); the enabled video
    models are the video daemon's engines (serve starts, restarts and stops it).
  - **The API**, behind `Authorization: Bearer <token>` (`NS_API_TOKEN`, else one made and kept in `serve.json`
    beside the registry; the page's API tab shows it): `POST /rpc/<kind>.<method>` - H3's rules: POST only, fixed
    routes, every variable in the JSON body, no path variables or query strings, `{"ok": true, "result": ...}` /
    `{"ok": false, "error": {"code", "message", "retryable", "status", "method", "field"}}` - **the body says
    everything**: branch on `error.code`, retry when `retryable`, `field` names the request key at fault, `status` is
    the code's number by convention. **Every /rpc answer is HTTP 200, errors too** (401, an unknown method, a body
    too large to read), so another transport can carry the same envelope; only the OpenAI routes and the files use
    HTTP statuses. An unreadable request (not POST, a query string, not a JSON object, over 256 MiB) is
    wrapped the same way (`invalid_request`, `request_too_large`); the studio's codes are kept for video (`busy`,
    `not_running` ...) and a failed job's `error` is the same object. `no_room` (503: the model's cards are held by
    other models now) is retryable; `conflict` (409: disabled, none enabled) is not - for `system` (state, busy, version, files.stat, files.delete, models, enable, disable, load, unload),
    `llm` (chat, models), `image` (generate, edit, status, jobs, cancel, queue.*, upload, canvases.list, info, progress,
    history), `audio` (generate, speak, status, jobs, cancel, queue.*, upload, voices.*, info, progress, history), `video` (the studio's: generate, status, wait, cancel {label}, queue.* - move and remove by label,
    generate's priority - jobs.*, timeline, upload ...), each kind with
    `describePromptGuideMD {model}`; and the OpenAI-compatible `/v1/models`, `/v1/chat/completions`,
    `/v1/images/generations|edits`, `/v1/audio/speech` and the files they link (and `/v1/video/files/<clip>`, `/v1/video/thumbs/<clip>.webp`). `GET /openapi.yml` describes the
    kinds that have an enabled model (enabling one adds its methods): every body key with its limits and names (the
    video engines as the daemon lists them), each answer's shape (glue/serve/src/home/spec.rs), each method's errors,
    and the video label convention ("Series NN/MM: text"); the API tab lists them with a curl each. The document
    needs no token and carries none (the page's copy fills it in).
    Pictures, songs and speech run as **jobs** (glue/serve/src/home/jobs.rs, kept in jobs.json beside serve.json):
    a label, `<kind>.status {label}`, `<kind>.cancel {label}` (the servers stop that request only, between steps),
    `wait: false` to answer at once; the call can drop and the job goes on, and a restart runs unfinished ones again.
    `reentrant_id` on every generate (video's too) makes a submit idempotent. A kind makes as many jobs at once as
    it has servers; the rest wait in a queue as video's (`priority`, `held`, `project`, `batch`, `queue: false`;
    `<kind>.queue.list|move|pause|resume`), and `stage` / `waiting_for` say what a job waits for. `system.busy` says where work runs
    (slots: two image and audio servers, the video daemon's cards) and each card's share; `system.version` lists the
    features by name; `system.files.stat|delete` give a file's bytes and SHA-256 or remove it (a clip with every
    file of it). video.generate also takes `frames` (exactly 17k + 5), a `chain` object (context_frames, sound,
    still, color_reanchor) and answers frames, seconds and the keys it `ignored` (negative_prompt: H3 runs at cfg 1).
    **API changes, 2026-10-15** (each also in `system.version`'s features):
    - Job requests are **checked when submitted** (glue/serve/src/home/precheck.rs): steps a few-step model does not
      run, size, sampler, cfg without a negative prompt, pictures that do not decode or pass 40 MP, a speech voice or
      language the model does not have (read from its config, loaded or not), a `continue_from` that is not here - a
      400 at once even with `wait: false`, not a failed job later.
    - `image.info` answers `steps` (`locked`, `allowed`: turbo runs only 6) and the limits (8 pictures, 40 MP each,
      256 MiB a request) without the model loaded.
    - `image.edit` `strength` (0.05-1): the denoise starts from the first picture noised that far (ComfyUI's denoise:
      the last steps of a round(steps / strength) run; a few-step model starts at strength itself). `reference:
      false` makes it a plain img2img (the pictures do not condition the model); `image.generate` takes `image` (an
      init picture) with `strength` for the same.
    - `audio.speak` `continue_from` (a speech file made here): its last sentences (up to 15 s, cut at a pause) are
      the in-context reference after the voice's sample, so the voice and its pace carry on between a book's chunks.
    - `<kind>.status` / `<kind>.cancel` find a job by `label` or, when that is absent or empty, `reentrant_id`;
      cancelling an ended job answers `already_ended` with its state, and not-found names what it looked up.
    - `system.busy.parallel`: how many jobs of each image / audio model run at once (one per card it is enabled on).
    - `video.status`: `preview` (a PNG of the shot so far after every step, from a fitted latent-to-RGB map),
      `failure` {code, retryable, message, hint} beside the text `error`, and `gpu`; `video.engines.list` inputs list
      the `ignored` keys.
    **API changes, 2026-10-15 (2), api 3** - for a client that also drives another engine box (nextcuda):
    - **/rpc errors are HTTP 200** with the envelope (before: the HTTP status mirrored `error.status`); a client
      reads `ok` and `error.code`. The OpenAI routes and file routes keep real statuses.
    - Image and audio results name their run details **`run`** (was `nextsycl`; the /v1 routes keep `nextsycl`).
    - `system.models` gives each model `capabilities` (neutral words: chat, tools, generate, edit, img2img, few_step,
      ocr, song, speech, clone, voice_design, text_to_video, reference_images ...) and the `methods` that take it.
    - `system.cards`: one neutral list of the GPUs (memory total / used / free, load, power, temperature, the models
      and processes on each, what the enabled models are counted for).
    - **`<kind>.status` answers one job** (label or reentrant_id needed); **`<kind>.jobs`** is the list (running,
      queued in run order, the last ended; `limit`, `project`).
    - **A queue for image and audio jobs**: as many run at once as the kind has servers (two); `priority`, `held`,
      `project`, `batch` and `queue` (false: 409 `busy`, retryable, when it would not start now) on submit;
      `<kind>.queue.list|move|pause|resume`; a waiting job is cancelled at once.
    - Jobs say **`stage`** (queued, held, starting, waiting, loading, running, then how they ended) and
      **`waiting_for`** in words (the jobs ahead, the model loading on GPU n, a card's room and who holds it, the
      job its server finishes first), plus `position`, `priority`, `project`, `batch`.
    - **Idle models make room at once** for a job that needs it (see the load rule above), and a job runs on the
      first of its cards that has (or can be given) the room; a job waits for room up to 30 minutes instead of
      failing after 90 s.
    - **Chat keep-up**: a chat request may send `X-Keep-Up-S: <seconds>` (/v1/chat/completions, through the front
      door or the switcher; empty or not a number = 300) or `keep_up_s` (llm.chat): the model is kept that long after
      the answer, and nothing stops it for room meanwhile. Without it the model may be preempted as soon as it is idle.
      Open WebUI sends it from its connection's headers (Admin Settings > Connections > the switcher's connection >
      Headers: `{"X-Keep-Up-S": "300"}`); the switcher's `GET /keep` lists what is kept.
    - **`llm.chat` answers 503 `no_room`** (retryable) like the other kinds when a chat model must load and its
      cards stay held, instead of a 15-minute timeout or a spill.
    - **Uploads**: `image.upload` / `audio.upload {name, data}` keep a picture or recording once (audio as WAV);
      `image`, `images`, `ref_audio` and voices.add's `sample` then take `upload:<name>`. `system.files.stat|delete`
      take their names.
    - Saved voices take an **`owner`** (voices.add / voices.design; voices.list filters by it; replacing or removing
      an owned voice must name it) and voices.list says whether each is **`ready`** and which enabled models speak it.
    - `continue_from` naming a file that is not here is 404 `not_found` (field continue_from), as the spec now says,
      the same as video's missing files and missing uploads.
    `/api/...` is the page's own.
  - **The settings** (the Config tab; `system.config`, `system.configSet {set: {NAME: value | null}} | {text}`,
    `system.restart {service}`): `nextsycl.conf` as every setting the program reads - what it does, its default,
    what reads it - or as the file itself. A save keeps the file's comments and order, keeps a copy
    (`nextsycl.conf.bak-<time>`, the last 10), and lists what has to restart to see the change, a button each (this
    page, the switcher, the studio, the video daemon, the image and music servers, the chat model). A setting in a
    service's environment wins over the file and is marked.
  - **Maintenance mode** (`nextsycl maintenance on --message TEXT --eta MINUTES [--unload]` / `off` / `status`, the
    page's Maintenance button, `system.maintenance {on, message, eta_s, unload}`): the box set aside for development
    while callers are told. What would start work - `llm.chat`, image / audio / video generation, edits, speech,
    `system.load` / `enable`, the OpenAI POST routes - is refused with error code `maintenance` (503, retryable,
    `retry_after_s`, the notice in `error.maintenance`; `/v1` answers HTTP 503 with `Retry-After`). Status, listings,
    cancels and guides answer as always, and `system.busy` / `version` / `state` carry the notice. Queued image and
    audio jobs wait (stage `maintenance`); the video queue pauses (what was paused before is put back when it ends);
    running work finishes; `--unload` unloads the models not answering. Open WebUI's chats are answered with the
    notice by the switcher (it reads `maintenance.json` beside the registry). Turning it on prints a bypass key:
    requests with `X-Maintenance-Bypass: <key>` go through, so a developer can test through :8000 meanwhile.

  ![nextsycl serve: the GPUs, the models](screenshots/home-desktop.webp)
  ![nextsycl serve: the API tab](screenshots/home-api.webp)
  ![nextsycl serve: the settings](screenshots/home-config.webp)

- **`switch`**: one OpenAI endpoint for a chat front end over the one model server that runs at a time. It lists the
  registry's enabled chat models (read on every request); a request naming another model asks the video studio to
  swap (`llm.mode`: it knows when a render holds a card), waits until the server answers as that model, and passes the
  request through, streamed answers included. The loaded model is not swapped out while it answers or within 90 s of
  its last request (that request gets a 409 naming it). Per entry: `tools: false` drops tool definitions, `tasks:
  false` declines a front end's background tasks. `--alias` keeps old model ids working. `--images`: the images API
  (`/v1/images/...`) goes to the image server and its model is listed while it runs - one base URL for chat and
  pictures. In maintenance mode it answers chat requests with the notice (503, `Retry-After`) and loads nothing.
- **`gpustat`**: the card's telemetry every few seconds into a JSON file the pages read: VRAM used (the DRM clients'
  `drm-resident-vram0` in /proc/*/fdinfo - root only) and total, busy %, power over a 15 s window, the power cap,
  temperatures, fan, the PCIe link it trained at and what the card and slot could do.

Example units: `docs/host/*.service.example`.
