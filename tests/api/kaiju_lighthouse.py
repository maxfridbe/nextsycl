#!/usr/bin/env python3
"""Reference test: a 30 s film with sound made through the front door's API alone, joined here with ffmpeg.

A painting (image.generate, or --picture) is the first frame of three chained clips (video.generate): a giant kaiju
rises from the sea, wades to the cliff and knocks the lighthouse over, every frame kept in the painting's style.
Each later clip carries on the one before it - its motion (guide_clip: that clip's last frames as a moving keyframe
at frame 0) and its sound (first_audio: that clip's sound tail) - and so begins with that much of it again, and runs
that much longer. The finished clips are fetched with GET /v1/video/files/<file> and joined here: the shared part
cut in the middle for the picture and cross-faded for the sound. Then it checks what it made: the clips' sizes,
lengths and sound, the film's length, that clip 1 starts on the painting and that each clip begins with the last
one's end.

    tests/api/kaiju_lighthouse.py                      # the whole film: 3 x 10 s at 768x768, upscaled to 1536x1536
    tests/api/kaiju_lighthouse.py --quick              # a smoke run: 3 x 4 s at 384x384, 8 steps, no upscale
    tests/api/kaiju_lighthouse.py --url http://box:8000 --token ns-... --picture lighthouse.png --out out/kaiju
    tests/api/kaiju_lighthouse.py --reuse h3_20261011_144329.mp4 ...   # clip 1 already made: from clip 2 on

Needs: Python 3.9+, an ffmpeg and ffprobe with H.264 (not Fedora's ffmpeg-free; --ffmpeg / --ffprobe name
another); on the server an enabled video model (minimax-h3) and, without --picture, an image model. The token: --token, else the server's serve.json on this machine. Writes into --out: the picture,
the clips, film.mp4 and report.json. Exits 1 when a check fails.
"""

import argparse
import base64
import json
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

STYLE = ("An animated oil painting in the exact style of <Picture 1>: thick impressionist brushstrokes, visible "
         "canvas texture, warm golden dusk light, painterly clouds; every frame stays a moving painting.")
KAIJU = "the colossal dark-green reptilian kaiju, with jagged pale dorsal plates along its spine"   # --creature

# (the beat's name, what happens in it, its sound, its music) - one clip each, chained
BEATS = [
    ("rise",
     f"A wide shot of the red-and-white lighthouse on the rocky cliff at dusk, its lamp glowing, waves rolling "
     f"against the rocks. The calm sea begins to churn and bubble far out on the left. At 00:03.000 {{kaiju}} slowly "
     f"rises out of the ocean, seawater pouring off its scaly head and shoulders, until it stands chest-deep in the "
     f"surf and turns its head toward the lighthouse. The camera is static with a slight handheld shake.",
     "Waves crashing on rocks, gulls calling, then deep rumbling water surging and a massive splash.",
     "A low ominous orchestral drone with slow timpani hits that build in tension."),
    ("approach",
     f"Continuing the shot, {{kaiju}} lets out a thunderous roar, its dorsal plates glowing faintly blue, then wades "
     f"toward the cliff with heavy steps, huge waves breaking around its legs. At 00:06.000 it reaches the rocks "
     f"beside the lighthouse and raises a massive clawed arm high above the tower. The camera is static with a "
     f"handheld shake at each step.",
     "An enormous earth-shaking roar echoing over the sea, heavy sloshing footsteps, waves slamming the cliff.",
     "Rising orchestral strings and brass over pounding timpani."),
    ("topple",
     f"Continuing the shot, {{kaiju}} swings its clawed arm into the lighthouse. At 00:01.500 the tower cracks in the "
     f"middle, the glowing lamp room shatters into sparks and glass, and the whole lighthouse topples sideways and "
     f"crashes down the cliff into the sea, a huge spray of white water rising. At 00:06.000 the kaiju throws its "
     f"head back and roars at the dusk sky as dust and spray drift across the painted clouds. The camera is static "
     f"with a strong handheld shake on the impact.",
     "A crushing impact, stone cracking and crumbling, glass shattering, a huge splash, then a long roar.",
     "A loud orchestral climax with pounding timpani and brass, then a deep sustained low chord."),
]


def snap(frames, nearest=False):
    """A clip's frame count as the engine makes it: 17k + 5 (up from the count, or the nearest)"""
    n = max(5, round(frames))
    up = n + (5 - n) % 17
    return up if not nearest or up - n <= 8 or up - 17 < 5 else up - 17


def prompt_of(beat, first):
    """A beat's prompt: the first clip's names the painting (<Picture 1>, its first frame); a later one starts
    from the last clip's moving frames, which carry the style, and names no picture"""
    _, action, sound, music = beat
    action = action.format(kaiju=CREATURE)
    style = STYLE if first else STYLE.replace(" in the exact style of <Picture 1>", "")
    return (f"integrated_multimodal_description: [Shot 1] {style} {action}\n\n"
            f"overall_soundscape: {sound}\n\nnon_diegetic_music: {music}")


class Api:
    def __init__(self, url, token):
        self.url, self.token = url.rstrip("/"), token

    def _req(self, method, path, body=None, timeout=3600):
        data = None if body is None else json.dumps(body).encode()
        r = urllib.request.Request(self.url + path, data=data, method=method,
                                   headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(r, timeout=timeout) as a:
                return a.read()
        except urllib.error.HTTPError as e:
            if path.startswith("/rpc/"):
                return e.read()   # the envelope says what went wrong
            raise RuntimeError(f"{method} {path}: {e.code} {e.read()[:300]!r}") from None

    def rpc(self, method, body=None, timeout=3600):
        v = json.loads(self._req("POST", f"/rpc/{method}", body or {}, timeout))
        if not v.get("ok"):
            raise RuntimeError(f"{method}: {v.get('error', {}).get('message', v)}")
        return v["result"]

    def get(self, path, dst):
        Path(dst).write_bytes(self._req("GET", path))
        return dst


def token_of(arg):
    if arg:
        return arg
    for f in [Path.home() / ".config/nextsycl/serve.json", Path("/var/lib/nextsycl/serve.json")]:
        try:
            return json.loads(f.read_text())["token"]
        except (OSError, KeyError, ValueError):
            pass
    sys.exit("no token: --token (the front door's API tab)")


FFMPEG, FFPROBE = ["ffmpeg"], ["ffprobe"]
CREATURE = KAIJU


def ffmpeg_can():
    """What the join and the checks need: H.264 decoded (the clips) and encoded (the film, libx264)"""
    try:
        dec = subprocess.run([*FFMPEG, "-hide_banner", "-decoders"], capture_output=True, text=True).stdout
        enc = subprocess.run([*FFMPEG, "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
        subprocess.run([*FFPROBE, "-version"], capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError) as e:
        sys.exit(f"ffmpeg / ffprobe: {e}")
    if " h264 " not in dec or "libx264" not in enc:
        sys.exit(f"{shlex.join(FFMPEG)} cannot decode and encode H.264 (Fedora's ffmpeg-free cannot: RPM Fusion's ffmpeg can), "
                 "so it cannot read the clips or write the film - run this on another machine, or --ffmpeg / --ffprobe "
                 "naming one that can (a container's)")


def probe(f):
    o = subprocess.run([*FFPROBE, "-v", "error", "-show_entries", "stream=codec_type,width,height,nb_frames:format=duration",
                        "-of", "json", str(f)], capture_output=True, text=True, check=True)
    v = json.loads(o.stdout)
    vid = next((s for s in v["streams"] if s["codec_type"] == "video"), {})
    return {"width": vid.get("width"), "height": vid.get("height"), "frames": int(vid.get("nb_frames") or 0),
            "seconds": float(v["format"]["duration"]), "audio": any(s["codec_type"] == "audio" for s in v["streams"])}


def thumb(f, last=False, at=0, n=32):
    """A frame of a picture or clip as n x n grey values: the last, or frame `at`"""
    pre = ["-sseof", "-0.5"] if last else []
    sel = "" if last else rf"select=eq(n\,{at}),"
    o = subprocess.run([*FFMPEG, "-v", "error", *pre, "-i", str(f), "-vf", f"{sel}scale={n}:{n},format=gray", "-frames:v", "1" if not last else "999",
                        "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    return o[-n * n:] if last else o[:n * n]


def corr(a, b):
    """Pearson correlation of two frames' grey values: 1 the same picture, ~0 unrelated"""
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va, vb = sum((x - ma) ** 2 for x in a), sum((y - mb) ** 2 for y in b)
    return cov / ((va * vb) ** 0.5 or 1.0)


def wait_clip(api, project, n, label):
    """Polls the project's timeline until clip n is done (its file's name) or gone (failed: the studio's error)"""
    seen, gone, last = False, 0, ""
    while True:
        clips = api.rpc("video.timeline", {"project": project})["clips"]
        c = next((c for c in clips if c["n"] == n), None)
        if c and c["state"] == "done":
            return c["name"] + ".mp4"
        st = api.rpc("video.status")
        if c:
            seen, gone = True, 0
            if st.get("job", {}).get("label") == label:
                line = f"  {label}: {st.get('stage')} {st.get('pct')}%, {st.get('eta') or 0:.0f} s left"
                if line != last:
                    print(line, flush=True)
                    last = line
        else:
            # gone: failed (the status says so), or finished and its record not yet written (a minute's grace)
            gone += 1
            ours = st.get("job", {}).get("label") == label
            if (seen and ours and st.get("error")) or gone >= 6:
                err = st.get("error") if ours else "it left the queue"
                api.rpc("video.queue.clear", {"project": project})
                raise RuntimeError(f"{label} failed: {(err or '').splitlines()[0] if err else '?'} "
                                   f"(the studio pauses its queue after a failure: video.queue.resume)")
        time.sleep(10)


def loudness(f):
    """A file's integrated loudness (LUFS), or None for silence"""
    e = subprocess.run([*FFMPEG, "-hide_banner", "-vn", "-i", str(f), "-af", "ebur128=framelog=quiet", "-f", "null", "-"],
                       capture_output=True, text=True).stderr
    try:
        v = float(e[e.rindex("I:") + 2:].split()[0])
    except ValueError:
        return None
    return v if v > -70 else None


def join(clips, out, overlap, fps, lufs=-18.0, fade=1.5):
    """The clips into one film: each later clip begins with the last `overlap` frames of the one before - the
    picture cut in the middle of them, the sound cross-faded over all of them. Then the film's sound levelled to
    `lufs` as a whole (each clip levelled alone would flatten a score that builds) under a limiter, and both faded
    out over the last `fade` seconds."""
    frames = [probe(c)["frames"] for c in clips]
    n, head, tail = len(clips), overlap - overlap // 2, overlap // 2
    fc = []
    for i, nf in enumerate(frames):
        fc.append(f"[{i}:v]trim=start_frame={head if i else 0}:end_frame={nf - tail if i + 1 < n else nf},setpts=PTS-STARTPTS[v{i}]")
        fc.append(f"[{i}:a]apad,atrim=end={nf / fps:.6f},asetpts=PTS-STARTPTS[a{i}]")
    fc.append("".join(f"[v{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0[v]")
    prev = "a0"
    for i in range(1, n):
        nxt = "a" if i + 1 == n else f"x{i}"
        fc.append(f"[{prev}][a{i}]acrossfade=d={overlap / fps:.6f}:c1=tri:c2=tri[{nxt}]")
        prev = nxt
    if n == 1:
        fc.append("[a0]anull[a]")
    total = sum(frames) - overlap * (n - 1)
    if fade > 0:
        fc.append(f"[v]fade=t=out:st={max(0.0, total / fps - fade):.3f}:d={fade}[vf]")
    part = Path(out).with_suffix(".part.mp4")
    cmd = [*FFMPEG, "-y", "-v", "error"]
    for c in clips:
        cmd += ["-i", str(c)]
    cmd += ["-filter_complex", ";".join(fc), "-map", "[vf]" if fade > 0 else "[v]", "-map", "[a]", "-c:v", "libx264", "-crf", "18",
            "-preset", "medium", "-pix_fmt", "yuv420p", "-r", str(fps), "-c:a", "pcm_s16le", "-f", "mov", str(part)]
    subprocess.run(cmd, check=True)
    af = []
    if lufs and (l := loudness(part)) is not None:
        af.append(f"volume={lufs - l:.2f}dB")
    af.append("alimiter=limit=0.891:level=disabled")
    if fade > 0:
        af.append(f"afade=t=out:st={max(0.0, total / fps - fade):.3f}:d={fade}")
    subprocess.run([*FFMPEG, "-y", "-v", "error", "-i", str(part), "-c:v", "copy", "-af", ",".join(af), "-c:a", "aac", "-b:a", "192k",
                    "-movflags", "+faststart", str(out)], check=True)
    part.unlink()
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--token")
    ap.add_argument("--out", default="kaiju-lighthouse")
    ap.add_argument("--picture", help="the painting to start from (else image.generate makes one)")
    ap.add_argument("--image-model", default="qwen-image-2.1-q8")
    ap.add_argument("--quick", action="store_true", help="3 x 4 s at 384x384, 8 steps, no upscale")
    ap.add_argument("--seconds", type=float, help="each clip's (default 10; 768x768 for 15 s does not fit a 32 GB card)")
    ap.add_argument("--size", type=int, help="the square canvas (default 768)")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--upscale", type=float, help="default 2 (ESRGAN general x4v3 on the decoded frames)")
    ap.add_argument("--seed", type=int, default=1954)
    ap.add_argument("--creature", default=KAIJU, help="who rises from the sea, as the prompts name it")
    ap.add_argument("--engine", default="Q6_K", help="Q6_K: INT8's composition in 4 GiB less - room for the later clips' 10.92 s at 768x768, which INT8 runs out of memory on")
    ap.add_argument("--ffmpeg", default="ffmpeg", help="the ffmpeg command (words split as a shell would)")
    ap.add_argument("--ffprobe", default="ffprobe")
    ap.add_argument("--guide-frames", type=int, default=22, help="the last clip's frames a clip carries on (17k + 5: 5, 22, 39 ...)")
    ap.add_argument("--lufs", type=float, default=-18.0, help="the film's loudness (0: as made)")
    ap.add_argument("--fade-out", type=float, default=1.5, help="seconds")
    ap.add_argument("--reuse", default="", help="finished clips (file names, comma-separated) for the first beats")
    a = ap.parse_args()
    global FFMPEG, FFPROBE, CREATURE
    FFMPEG, FFPROBE = shlex.split(a.ffmpeg), shlex.split(a.ffprobe)
    CREATURE = a.creature
    ffmpeg_can()
    q = a.quick
    seconds = a.seconds or (4.0 if q else 10.0)
    size = a.size or (384 if q else 768)
    steps = a.steps or (8 if q else 12)
    upscale = a.upscale if a.upscale is not None else (1.0 if q else 2.0)
    fps, g = 24, a.guide_frames
    reuse = [r.strip() for r in a.reuse.split(",") if r.strip()]

    api = Api(a.url, token_of(a.token))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    run = time.strftime("%Y%m%d-%H%M%S")
    project, prefix = f"kaiju-lighthouse {run}", f"Kaiju {run}"
    report = {"url": a.url, "project": project, "settings": {"seconds": seconds, "size": size, "steps": steps, "upscale": upscale,
              "seed": a.seed, "engine": a.engine,
              "guide_frames": g}, "clips": [], "checks": {}}
    t0 = time.time()

    # 1. the painting
    if a.picture:
        pic = Path(a.picture)
    else:
        print(f"picture: {a.image_model}, {size}x{size}", flush=True)
        r = api.rpc("image.generate", {"model": a.image_model, "prompt": "a lighthouse at dusk, oil painting",
                                       "size": f"{size}x{size}", "steps": 20, "seed": a.seed})
        url = r["data"][0]["url"]
        pic = api.get(url[url.index("/v1/"):], out / "painting.png")
        api.rpc("system.unload", {"model": a.image_model})   # the card for the video model
    report["picture"] = str(pic)
    ref = api.rpc("video.upload", {"name": f"kaiju_{run}.png", "data": base64.b64encode(pic.read_bytes()).decode()})["name"]
    print(f"picture: {pic} -> {ref}", flush=True)

    # 2. the clips, each from the one before
    files, prev, lengths, film_frames, first_frames = [], None, [], 0, 0
    for i, beat in enumerate(BEATS, 1):
        label = f"{prefix} {i:02}/{len(BEATS):02}: {beat[0]}"
        # a clip is 17k + 5 frames; a later one begins with the last g frames of the one before, so it is that much
        # longer - each sized to bring the film to clip 1 + a beat a clip (nearest valid count)
        if prev is None:
            nf = snap(seconds * fps)
        else:
            nf = snap(first_frames + (i - 1) * seconds * fps - film_frames + g, nearest=True)
        secs = round(nf / fps, 4)
        lengths.append(secs)
        body = {"prompt": prompt_of(beat, prev is None), "seconds": secs, "steps": steps, "seed": a.seed + i - 1, "width": size,
                "height": size, "engine": a.engine, "project": project, "label": label, "queue": True}
        if prev is None:
            body["first_frame"] = ref
        else:
            body |= {"guide_clip": f"{prev}:{g}:0", "first_audio": prev, "first_audio_s": round(g / fps, 4)}
        if upscale > 1:
            body |= {"upscale": upscale, "upscaler": "esrgan-general"}
        t = time.time()
        if i <= len(reuse):
            name = reuse[i - 1]
            print(f"{label}: reusing {name}", flush=True)
        else:
            api.rpc("video.generate", body)
            print(f"{label}: queued ({'from ' + ref if prev is None else f'carrying on {prev}: {g} frames, {g / fps:.2f} s of sound'})", flush=True)
            name = wait_clip(api, project, i, label)
        dst = api.get(f"/v1/video/files/{name}", out / f"{i:02}_{beat[0]}.mp4")
        rec = api.rpc("video.jobs.get", {"name": name})
        p = probe(dst)
        report["clips"].append({"label": label, "file": name, "wall_s": round(time.time() - t, 1), "times": rec.get("times"),
                                "energy_wh": rec.get("energy_wh"), **p})
        print(f"{label}: {name} in {time.time() - t:.0f} s, {p['width']}x{p['height']}, {p['seconds']:.2f} s", flush=True)
        files.append(dst)
        film_frames += p["frames"] - (g if prev else 0)
        first_frames = first_frames or p["frames"]
        prev = name

    # 3. the film
    film = out / "film.mp4"
    nframes = join(files, film, g, fps, a.lufs, a.fade_out)
    fp = probe(film)
    report["film"] = {"file": str(film), "wall_s": round(time.time() - t0, 1), "lufs": loudness(film), **fp}
    print(f"film: {film}, {fp['width']}x{fp['height']}, {fp['seconds']:.2f} s", flush=True)

    # 4. the checks
    want = round(size * upscale)
    c = report["checks"]
    c["clips are the size asked"] = all(x["width"] == want and x["height"] == want for x in report["clips"])
    c["clips are the length asked"] = all(abs(x["seconds"] - want_s) < 0.5 for x, want_s in zip(report["clips"], lengths))
    c["clips have sound"] = all(x["audio"] for x in report["clips"])
    c["film is the clips joined"] = abs(fp["seconds"] - nframes / fps) < 0.25 and fp["audio"] and fp["width"] == want
    c["film is at its loudness"] = not a.lufs or (report["film"]["lufs"] is not None and abs(report["film"]["lufs"] - a.lufs) < 1.5)
    target = (first_frames + (len(BEATS) - 1) * seconds * fps) / fps
    c[f"film is {target:.1f} s (clip 1, then {seconds:g} s a beat)"] = abs(fp["seconds"] - target) < 0.4
    start = corr(thumb(pic), thumb(files[0]))
    c["clip 1 starts on the painting"] = start > 0.8
    # a later clip's frame g-1 is the last clip's last frame (its first g frames: the last clip's end, moving)
    joins = [corr(thumb(files[i], last=True), thumb(files[i + 1], at=g - 1)) for i in range(len(files) - 1)]
    c["each clip begins with the last one's end"] = all(j > 0.8 for j in joins)
    report["similarity"] = {"painting -> clip 1": round(start, 3), "joins": [round(j, 3) for j in joins],
                            "painting -> film's end": round(corr(thumb(pic), thumb(film, last=True)), 3)}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    for k, v in c.items():
        print(f"{'ok  ' if v else 'FAIL'} {k}")
    print(f"similarity {report['similarity']}; report {out / 'report.json'}")
    return 0 if all(c.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
