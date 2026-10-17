# B70 startup guide

From a fresh clone to chat, pictures, speech, songs and video on an Intel Arc Pro B70. Every step here was run as
written on 11 October 2026, on a fresh clone of `main` in an empty directory, with the times and outputs it gave:
a Fedora 44 box (kernel 7.2, 32 cores, 64 GB RAM) with an Arc Pro B70 and an Arc Pro B65. A B65 is optional;
everything but the two-card entries runs on one B70.

## 1. What the machine needs

- **Linux with the `xe` driver** bound to the card: `ls /dev/dri` lists a `renderD*` node per card, and
  `cat /sys/class/drm/card*/device/uevent | grep DRIVER` says `DRIVER=xe`. The oneAPI runtime, the compiler and the
  libraries all live in a container image, so the host needs no Intel software.
- **podman** (5.x here) with **crun** (rootless podman reaches `/dev/dri` only through crun's keep-groups), or docker.
- **Disk**: ~10 GB for the build image (and ~9 GB for its base image, pulled once), ~2 GB for the build, and the
  models (13-90 GB each; `nextsycl models search` lists sizes).
- **Your user in the `render` group** (`id` shows it), so the container can open the card.

## 2. Get it and build it

```sh
git clone https://github.com/maxfridbe/nextsycl.git
cd nextsycl
./build.sh image      # the build image, localhost/nextsycl-build - 10 min (plus the 9 GB base image's download)
./build.sh            # the kernel libraries, the program and the web pages, into dist/ - 5.5 min
./build.sh test       # the lints, the unit tests, the architecture's rules - 12 s after a build
```

Instead of building the image, you can take the one every push to `main` publishes (the same Containerfile, tagged
by its hash):

```sh
ref=ghcr.io/maxfridbe/nextsycl/nextsycl-build:$(sha256sum container/Containerfile | cut -c1-12)
podman pull $ref && podman tag $ref localhost/nextsycl-build
```

`./build.sh` alone builds the image first when it is missing. What `dist/` holds afterwards: `nextsycl` (the
program), `libnextsycl-{llm,image,video,audio}.so` (one kernel library a kind), `libnextsycl-flash.so` and
`libdnnl.so.3` (the diffusion kernels' attention), `silo/libnextsycl-qwen35.so` (Qwen3.8-27B's own tuned kernels)
and `wfe/` (the web pages). The build prints ~90 compiler notes about register spills in a few kernels; they are
expected.

## 3. Settings

One file beside the repository, `nextsycl.conf` (`NAME=value` lines; the environment wins over it):

```sh
echo "NS_MODELS=$HOME/models" > nextsycl.conf     # where model files go (and are seen as /models in the containers)
```

That is the only setting needed. Every other one, with its default, is in [docs/environment.md](docs/environment.md);
the front door's Config tab edits this file. The registry of installed models is `~/.config/nextsycl/models.json`.

## 4. Check the card

```sh
$ dist/nextsycl gpus
GPU 0: Intel(R) Arc(TM) Pro B70 Graphics, 31.9 GiB (31.6 GiB free), 256 compute units at 2800 MHz
GPU 1: Intel(R) Arc(TM) Pro B65 Graphics, 31.9 GiB (31.7 GiB free), 160 compute units at 2400 MHz
GPU 0: 256 MiB to the GPU 26.7 GB/s, back 28.3 GB/s, identical
...
$ dist/nextsycl llm selftest      # and image / video / audio selftest
llm: libnextsycl-llm.so on Intel(R) Arc(TM) Pro B70 Graphics (GPU 0): [1, 2, 3, 4] x 2 = [2.0, 4.0, 6.0, 8.0] - ok
```

Commands that run the kernels in the program itself (`gpus`, `llm generate`, `spec-check`, the selftests, `video
job ... --here`) start themselves inside the build image when the host has no oneAPI - nothing to set up.

## 5. Models

The models nextsycl offers are model cards (`model-cards/<kind>/<id>.yml`: the files with their links and
checksums, the settings, the silo, the prompt guide - [model-cards/README.md](model-cards/README.md)):

```sh
dist/nextsycl models search                 # everything; --kind llm|image|video|audio|lora
dist/nextsycl models pull qwen3.8-27b-q4    # downloads (resumable), checks SHA-256, registers
dist/nextsycl models list
```

Good first models for one B70:

| Model | Kind | Size | What |
|---|---|---:|---|
| `qwen3.8-27b-q4` | chat | 16.6 GB | Qwen3.8-27B 4-bit with its MTP draft block: ~57 tok/s, 128K + 32K context |
| `qwen-image-2.1-q8` | image | 16.4 GB | text to image (RGBA), edits |
| `qwen3-tts-custom` | speech | 4.2 GB | nine voices, styled by an instruction, 10 languages |
| `voxcpm2` | speech | 4.6 GB | 30 languages at 48 kHz, voices designed from a description or cloned |
| `breeze-tts-2` | speech | 3.4 GB | English and Chinese: voices designed, cloned or directed, vocal events (non-commercial license) |
| `vibevoice-1.5b` | speech | 4.2 GB | long-form scripts of up to four speakers, each voice cloned from a recording (MIT; no impersonation) |
| `minimax-music3` | music | 26.6 GB | songs from lyrics and a description |
| `minimax-h3` | video | 47.9 GB | text / pictures / a voice to video with sound |

Files you already have elsewhere on the machine are linked, not downloaded again: `models pull ID --from DIR`
(searched by name and size; a hard link on the same filesystem, else a symbolic link with the registry naming the
real file). A model pulled in another form (`qwen3.8-27b-q4-256k` beside `qwen3.8-27b-q4`) links the files the
registry already has.

## 6. Chat

```sh
dist/nextsycl llm start qwen3.8-27b-q4      # loads in ~9 s once the file is in the page cache; API on 127.0.0.1:8085
dist/nextsycl llm chat "Write a haiku about a lighthouse." --max 1000
dist/nextsycl llm ps                        # the requests: tokens, tok/s, energy
dist/nextsycl llm status                    # live: the card's memory, temperatures, power
dist/nextsycl llm stop
```

What it gave: 300 tokens at 50 tok/s, 255 W on the B70; through the OpenAI API
(`curl 127.0.0.1:8085/v1/chat/completions -d '{"model": "qwen3.8-27b-q4", "messages": [...]}'`) 65 tok/s. The model
thinks before it answers (`--effort low|high|max`): give it room (`--max 1000` or more) or the answer is cut off
inside its reasoning.

## 7. The front door: one page and one API for everything

```sh
dist/nextsycl serve                          # http://<this machine>:8000/
```

The page shows the GPUs and every model (a pulled model is enabled - offered - until you disable it). The API's bearer token is on its API tab
(and in `~/.config/nextsycl/serve.json`); `GET /openapi.yml` describes every method. A model loads on its first
request and unloads after `NS_IDLE_MINUTES` (10) without one. Run on this box:

```sh
T=<the token>; H="Authorization: Bearer $T"
curl -s localhost:8000/rpc/llm.chat -H "$H" -d '{"model": "qwen3.8-27b-q4", "messages": [{"role": "user", "content": "Say hello in five words."}], "max_tokens": 600}'
curl -s localhost:8000/v1/chat/completions -H "$H" -d '{"model": "qwen3.8-27b-q4", "messages": [...], "stream": true}'
curl -s localhost:8000/rpc/image.generate -H "$H" -d '{"model": "qwen-image-2.1-q8", "prompt": "a lighthouse at dusk, oil painting", "size": "512x512", "steps": 20}'
curl -s localhost:8000/v1/audio/speech -H "$H" -d '{"model": "qwen3-tts-custom", "input": "Good morning.", "voice": "ryan", "language": "english"}' -o hello.wav
curl -s localhost:8000/rpc/audio.generate -H "$H" -d '{"model": "minimax-music3", "instructions": "warm acoustic folk, 90 BPM", "input": "[verse]\nRiver runs slow tonight\n[chorus]\nCarry me home", "seconds": 15}'
curl -s localhost:8000/rpc/llm.describePromptGuideMD -H "$H" -d '{"model": "qwen3.8-27b-q4"}'   # how to prompt it
```

| Request | First (loads the model) | Generating |
|---|---:|---:|
| chat, 5 words | 8 s | ~60 tok/s |
| picture 512x512, 20 steps | 12 s | 3.2 s |
| speech (Qwen3-TTS, VoxCPM2), 3 s of audio | 13-14 s | - |
| song, 15 s | 88 s | 29.5 s |

A request naming another model of the same kind swaps it in (after the request running). A card keeps 1.5 GiB free:
past its memory the xe driver spills into host memory and can hang the machine, so a model that does not fit waits
for the card (or answers 409) rather than loading anyway.

## 8. Video

```sh
dist/nextsycl video job generate --model minimax-h3 --prompt "a lighthouse on a cliff at dusk, waves crashing, gulls calling" \
    --width 384 --height 288 --seconds 2 --steps 8 --out clips/lighthouse.mp4 --gpu 0 --here
```

26 s for that clip (12 s of it the text encoder), H.264 with AAC sound, beside its last frame and latents. For a
queue, projects and the studio's web page: `nextsycl video start` (the daemon) and `nextsycl video serve` (README
"Video").

## 9. Both cards (a B70 and a B65)

The `-256k` entries (`qwen3.8-27b-q4-256k`, `qwen3.8-27b-q8-256k`; `gpus: all`) split the layers: the B70 (the
faster card) takes all it can beside the head, the draft block and the caches, the B65 the rest.

```
$ dist/nextsycl llm start qwen3.8-27b-q4-256k
1    Arc Pro B65         9.6GiB / 31.9GiB   22.3GiB     0-15
0    Arc Pro B70        29.0GiB / 31.9GiB    2.9GiB    16-63
```

It is for room (a 256K context, the 8-bit file with long contexts), not speed: ~52 tok/s against one B70's ~57.

## 10. Is it right? Is it as fast?

```sh
dist/nextsycl llm spec-check ~/models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_M.gguf --prompt "Explain why the sea is salty." --rows 4
# worst max|diff| / max|ref| 0.00e0; top-1 differs in 0 of 116 rows (4-row passes)
dist/nextsycl llm generate ~/models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_M.gguf --prompt-file PROMPT --max 128 --gpu 0
```

`spec-check` holds the drafted (MTP) passes to the one-token decode: 0 difference. On the benchy 2,185-token prompt
this build read 1,182-1,188 tok/s and decoded at 57.3-58.0 tok/s, the README's numbers; runs vary by ~4%.

## When something is off

| What you see | Why, and what to do |
|---|---|
| `the image localhost/nextsycl-build is not there` | `./build.sh image` (or set `NS_IMAGE` to another image with oneAPI) |
| `libmkl_sycl_blas.so.6: cannot open shared object file` | an old build: `git pull && ./build.sh` (GPU commands now start themselves in the image) |
| `crun is not installed` | install crun: rootless podman needs it to hand the container the card |
| `GPU 0 is busy: ... needs ~N GiB` | another model holds the card: it unloads when idle, or `system.unload` / `nextsycl <kind> stop` |
| `the model's silo is not built` | `./build.sh kernels`: the 27B's cards require its tuned kernels (the shared ones are 2-4x slower) |
| a chat answer that stops mid-thought | raise `max_tokens`: the model reasons first (`--effort low` reasons least) |
| an installed model not in `models list` | `NS_REGISTRY` names another registry, or the entry is in `~/.config/nextsycl/models.json` of another user |

One install a machine: the servers are containers with fixed names (`nextsycl`, `nextsycl-image`, `nextsycl-audio`,
`nextsycl-video`), so a second install on the same box (a checkout to try something) must not start them while the
first one's run.
