# API reference tests

Scripts that use a running front door (`nextsycl serve`, :8000) the way a client would: only its API, with the
bearer token. Each is a worked example of a whole job and a test of it: it checks what it made and exits 1 when a
check fails. They need Python 3.9+ (no packages) and, where they say so, ffmpeg.

| Script | What it makes | Through | Time on a B70 |
|---|---|---|---|
| `kaiju_lighthouse.py` | a 30 s film with sound: a painted lighthouse (or `--picture`) knocked over by a kaiju rising from the sea, in the painting's style - three clips at 768x768 (each carrying on the last: its last 22 frames and their sound), upscaled to 1536x1536, joined here with ffmpeg (sound to -18 LUFS, 1.5 s fade) | `image.generate`, `system.unload`, `video.upload`, `video.generate`, `video.timeline`, `video.status`, `video.jobs.get`, `GET /v1/video/files/<file>` | ~30 min (each later clip ~9.5 min on Q6_K, 12 steps); `--quick` ~4 min |

```sh
tests/api/kaiju_lighthouse.py --quick                       # the smoke run
tests/api/kaiju_lighthouse.py --picture lighthouse.png      # the film, from your own painting
tests/api/kaiju_lighthouse.py --url http://box:8000 --token ns-...
```

The token is `--token`, else the server's `serve.json` on the same machine (the page's API tab shows it).
Each run is its own video project (`kaiju-lighthouse <time>`), so it can be watched in the studio and run again.
