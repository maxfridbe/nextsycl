# Models, drives and engine options

[← nextsycl](../README.md)

The models a machine serves are entries in one registry (`NS_REGISTRY`, a JSON file; default
`~/.config/nextsycl/models.json`). An entry has the GGUF file (its first shard), the GPUs, the session contexts, the
engine settings (`NS_QW_MTP`, `NS_QW_CVEC` and the like), whether it is offered, and what a client may send it
(tools, background tasks). Its id is the model id clients see and the name the server answers under;
`nextsycl llm start <id>` runs it.

```sh
nextsycl models list [--json]
nextsycl models add <id> <file.gguf> [--title T] [--gpu 0[,1] | all] [--ctx N[,M...]] [--set NAME=VALUE]...
                    [--no-tools] [--no-tasks] [--disabled]
nextsycl models search [TEXT] [--kind llm|image|video|audio|lora]   # the catalog of supported models
nextsycl models pull <id>... [--dir DIR] [--from DIR]... [--verify] [--again]   # download (resumable), SHA-256, register
nextsycl models remove <id> [--files]        # --files deletes the GGUF shards too
nextsycl models enable <id> | disable <id>
nextsycl llm start <id>
nextsycl llm serve <id>                      # the same in the foreground, its log here
```

```
$ nextsycl models list
ID                                           STATE        SIZE  GPUS     CONTEXT          TITLE
glm-5.3-flash-uncensored                     enabled     89.9G  all      65536            GLM-5.3-Flash uncensored IQ2 · 64K context (both GPUs)
glm-5.3-flash-uncensored-256k                enabled     89.9G  all      262144,32768     GLM-5.3-Flash uncensored IQ2 · 256K context + a 32K chat (both GPUs)
qwen3.8-flash-next-coder-iq1_m               enabled     54.4G  0        131072,32768     Qwen3.8-Flash-Next Coder IQ1_M (coding; the B70, 128K + 32K)
qwen3.8-flash-next-iq2_xs-uncensored         enabled     63.4G  0        131072,32768     Qwen3.8-Flash-Next IQ2_XS uncensored (refusal projection; the B70, 128K + 32K)
swift-1.5-iq2_xs                             enabled     63.5G  0        131072,32768     Swift 1.5 IQ2_XS (short thinking; the B70, 128K + 32K)
```

A model switcher in front of the server (one model serves at a time) can read the registry for its list: an entry
added or disabled shows at once. With `NS_STUDIO_MODES` set to a mode file (one mode a model, its start command
`nextsycl llm start <id>`), every registry change rewrites that file's nextsycl entries (`"managed_by": "nextsycl"`) and
leaves the others alone. `nextsycl llm start <id>` mounts the entry's files at the same paths in the container, so an
entry can point anywhere on the machine.

## Drives: where the files are, and moving them

The front door's **Drives** panel (and `system.drives`, `nextsycl drives`) lists every drive a registered model's files
sit on, plus the models folders set (`NS_MODELS`, `NS_MODEL_DIRS`): each filesystem once, with its device and PCIe
link (sysfs), mount, size, free space, the models' bytes on it, and its read speed once measured. **measure**
(`system.drive.measure`, `nextsycl drives measure <drive>`) reads up to 4 GiB of the drive's largest model files with
O_DIRECT - past the page cache, nothing written - with 4 reads of 16 MiB in flight, and keeps the result with its date
in `serve.json`; it is refused while a model loads from that drive or a move uses it.

Each model row shows where its files are and a **→ drive** button for every other drive; `system.move {model,
drive}` and `nextsycl models move <id> --to <drive | mount>` do the same (the command hands the move to the front door
when it is up, else runs it itself; Ctrl-C or `--cancel` stops it). A move:

1. is refused while the model is loaded, loading or held by the video daemon (stop or disable it first), while
   another move runs (one at a time: `moving.json` beside the registry), while any process has one of its files open
   or mapped, or when the drive lacks the room: the bytes to write plus 1% of the drive (at least 4 GiB) kept free;
2. copies each file - its file and the shards beside it, its files by role, settings that are paths (a folder with
   what is in it) - reading past the page cache and dropping the written pages as it goes (a model's worth of cache
   would push out what the loaded models use), then reads the copy back with O_DIRECT: the same size and SHA-256 or
   the copy is removed and the move fails. A second name of the same file (a hard link) is linked again there;
3. repoints the registry entry in one write (its `file`, `files` and path settings; `path:scale` and lists kept), so the
   next start mounts the new paths;
4. removes the old files - except those another registered model still names (those stay, and that model with them:
   the panel's result lists what was kept and why) - and the folders that emptied.

A cancel or a failure before step 3 removes what the move wrote; the registry stays as it was. Meanwhile nothing
loads the model: the front door's enable and load refuse, a job waits for the move as it waits for room, and
`nextsycl <kind> start` refuses.

**Where files go.** A drive's models folder is a folder set on it (`NS_MODELS`, `NS_MODEL_DIRS`), else
`<mount>/nextsycl-models`. A file below a models folder keeps its path below it (`~/models/olmocr2/x.gguf` ->
`/var/mnt/2TBSSD/nextsycl-models/olmocr2/x.gguf`, and a move back lands where it was); a file anywhere else keeps its
layout below its model's folder - the nearest folder its files share, under its own name
(`/var/mnt/2TBSSD/minimaxmusic3/models/minimax-music3/vocoder/...` -> `~/models/minimax-music3/vocoder/...`). A file
already at the target with the same bytes (a move that did not finish, or another model's copy) is used, not copied;
one with other bytes stops the move.

**Streaming models.** A model whose card says `streams: true` reads weights from its files while it serves
(deepseek-r1-iq1s and V4-Flash read experts from the file tier, the Qwen3.8-Flash-Next family its PLE rows every
window): its drive's speed is its speed, so the page asks before moving one to a slower drive (measured, else by
link), and the command wants `--yes`. Load-once models lose only load seconds on a slower drive.

```
$ nextsycl drives
DRIVE       DEVICE                     LINK          MOUNT                  SIZE     FREE    MODELS  READ
nvme0n1p3   SKHynix_HFS001TEJ9X162N    PCIe 4.0 x4   /var/home              951G      27G      596G  7.2 GB/s (2026-10-17)
            37 models: glm-5.3-flash-uncensored ... deepseek-r1-iq1s ... breeze-tts-2
nvme1n1     PCIe SSD                   PCIe 3.0 x4   /var/mnt/2TBSSD       1877G     344G      287G  1.7 GB/s (2026-10-17)
            20 models: qwen3.8-flash-next-coder-iq1_m ... qwen3.8-27b-q4-256k breeze-tts-2
```

(hardac1: the P41 measures 7.2 GB/s with 4 reads in flight - a single `dd` reads 6.3 -, the DRAM-less Phison 1.7.)

![the Drives panel during a move](screenshots/home-drives.webp)


# Engine options: `--opt-NAME`

Every engine declares the options it takes beyond its kind's common ones; any command forwards them to it:
`--opt-NAME VALUE` (or `--opt-NAME` alone for a switch), NAME the option or the variable it sets. At load they set
those variables (what the engines and their kernels read; into the container too), per request they ride in the
request (`"options": {NAME: value}` in an API body). An option the engine does not take is an error that lists those
it does; `nextsycl llm engines`, `nextsycl image engines`, `nextsycl audio engines` show them all.

```sh
nextsycl image gen "a fox" --opt-int8 --opt-sigmas 1,0.9375,0.875,0.75,0.5,0.25     # int8 DiT, a 6-step schedule
nextsycl llm start qwen3.8-flash-next-iq2_xs --opt-chunk 4096 --opt-spec 4
curl localhost:8086/v1/images/generations -d '{"prompt": "a fox", "options": {"sigmas": "1,0.75,0.5,0.25"}}'
```
