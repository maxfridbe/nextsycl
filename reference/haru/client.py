"""A /v1/realtime client for a lockstep model (Haru-4B; standard library only): streams a conversation's user side
at real time, a 133 ms frame at a time - recordings with silence between them, as a person on a call - and records
what comes back: every event, the model's frames back to back (model.wav), a stereo mix (you left, it right:
call.wav), its text with the time it came, and the timings that matter in lockstep (each frame's turnaround: from
sending a frame to the model's answer to it).

    python3 -I client.py --url ws://127.0.0.1:8087/v1/realtime --out DIR [--token T] [--voice SAVED] [--lang en]
                         [--card card.json] [--say q1.wav --gap 4 --say q2.wav --gap 6 ...] [--barge-at S --say q.wav]
                         [--lead 2] [--speak-first]

--say / --gap: the script in order (a recording, then seconds of quiet room noise); --lead: seconds of quiet before
the first; --barge-at S: the next --say starts S seconds after the model's text first appears instead of after its
--gap (speaking over it: does it stop?). Writes DIR/events.jsonl, model.wav, user.wav, call.wav, text.jsonl
(the model's text with times) and summary.json.
"""
import argparse
import base64
import json
import os
import random
import socket
import struct
import threading
import time
import wave
from urllib.parse import urlparse

FRAME = 3200
RATE = 24000


class Ws:
    def __init__(self, url, token=None):
        u = urlparse(url)
        self.s = socket.create_connection((u.hostname, u.port or 80))
        self.s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        key = base64.b64encode(os.urandom(16)).decode()
        path = u.path + (("?" + u.query) if u.query else "")
        head = (f"GET {path} HTTP/1.1\r\nHost: {u.hostname}:{u.port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n")
        if token:
            head += f"Authorization: Bearer {token}\r\n"
        self.s.sendall((head + "\r\n").encode())
        resp = b""
        while b"\r\n\r\n" not in resp:
            c = self.s.recv(1)
            if not c:
                raise RuntimeError("closed during the handshake: " + resp.decode(errors="replace"))
            resp += c
        if b" 101 " not in resp.split(b"\r\n")[0]:
            raise RuntimeError("no upgrade: " + (resp + self.s.recv(65536)).decode(errors="replace"))
        self.lock = threading.Lock()

    def send(self, obj):
        data = json.dumps(obj).encode()
        h = bytearray([0x81])
        n = len(data)
        if n < 126:
            h.append(0x80 | n)
        elif n < 65536:
            h.append(0x80 | 126)
            h += struct.pack(">H", n)
        else:
            h.append(0x80 | 127)
            h += struct.pack(">Q", n)
        mask = os.urandom(4)
        h += mask
        body = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
        with self.lock:
            self.s.sendall(bytes(h) + body)

    def _exact(self, n):
        b = b""
        while len(b) < n:
            c = self.s.recv(n - len(b))
            if not c:
                raise EOFError
            b += c
        return b

    def recv(self):
        data = b""
        while True:
            h = self._exact(2)
            fin, op, n = h[0] & 0x80, h[0] & 0x0F, h[1] & 0x7F
            if n == 126:
                n = struct.unpack(">H", self._exact(2))[0]
            elif n == 127:
                n = struct.unpack(">Q", self._exact(8))[0]
            p = self._exact(n)
            if op == 0x8:
                return None
            if op in (0x9, 0xA):
                continue
            data += p
            if fin:
                return data.decode(errors="replace")


def read_wav(path):
    """16-bit WAV -> mono 24 kHz int samples (linear resampling: test clips only)"""
    with wave.open(path, "rb") as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        assert sw == 2, "16-bit WAV expected"
        raw = w.readframes(w.getnframes())
    pcm = struct.unpack(f"<{len(raw) // 2}h", raw)
    if ch > 1:
        pcm = [sum(pcm[i:i + ch]) // ch for i in range(0, len(pcm), ch)]
    if sr != RATE:
        n = len(pcm) * RATE // sr
        out = []
        for i in range(n):
            t = i * sr / RATE
            j = int(t)
            f = t - j
            a, b = pcm[min(j, len(pcm) - 1)], pcm[min(j + 1, len(pcm) - 1)]
            out.append(int(a + (b - a) * f))
        pcm = out
    return list(pcm)


def write_wav(path, samples, channels=1):
    with wave.open(path, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(struct.pack(f"<{len(samples)}h", *[max(-32768, min(32767, int(x))) for x in samples]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--token")
    ap.add_argument("--out", required=True)
    ap.add_argument("--voice")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--card", help="a JSON file: the character card's fields")
    ap.add_argument("--say", action="append", default=[])
    ap.add_argument("--gap", action="append", type=float, default=[])
    ap.add_argument("--lead", type=float, default=2.0)
    ap.add_argument("--barge-at", type=float)
    ap.add_argument("--speak-first", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ws = Ws(a.url, a.token)
    t0 = time.time()
    log = open(os.path.join(a.out, "events.jsonl"), "w")
    st = {"model": bytearray(), "deltas": [], "text": [], "started": [], "stopped": [], "errors": [], "created": None, "session": None, "first_text": None}
    lock = threading.Lock()

    def reader():
        while True:
            try:
                m = ws.recv()
            except Exception:
                break
            if m is None:
                break
            now = time.time() - t0
            v = json.loads(m)
            t = v.get("type", "")
            slim = {k: (x if k != "delta" or not t.endswith("audio.delta") else f"<{len(x)} b64>") for k, x in v.items()}
            log.write(json.dumps({"t": round(now, 3), **slim}, ensure_ascii=False) + "\n")
            with lock:
                if t == "session.created":
                    st["session"] = v["session"]
                elif t == "response.created":
                    st["created"] = now
                elif t in ("response.audio.delta", "response.output_audio.delta"):
                    b = base64.b64decode(v["delta"])
                    st["model"].extend(b)
                    st["deltas"].append((now, len(b) // 2))
                elif t in ("response.audio_transcript.delta", "response.output_audio_transcript.delta"):
                    st["text"].append((round(now, 3), v["delta"]))
                    if v["delta"].strip() and st["first_text"] is None:
                        st["first_text"] = now
                elif t == "input_audio_buffer.speech_started":
                    st["started"].append(now)
                elif t == "input_audio_buffer.speech_stopped":
                    st["stopped"].append(now)
                elif t == "error":
                    st["errors"].append(v["error"])

    threading.Thread(target=reader, daemon=True).start()
    session = {"modalities": ["audio", "text"], "input_audio_format": "pcm16", "output_audio_format": "pcm16",
               "nextsycl": {"language": a.lang, "input_sample_rate": RATE, "seed": a.seed, "options": {}}}
    if a.voice:
        session["voice"] = a.voice
    if a.card:
        session["nextsycl"]["card"] = json.load(open(a.card))
    if a.speak_first:
        session["nextsycl"]["options"]["speak-first"] = "1"
    ws.send({"type": "session.update", "session": session})

    # the user's side: lead quiet, then each recording and its gap of quiet room noise
    user = []
    sent_at = []

    def quiet(secs):
        return [int(random.gauss(0, 6)) for _ in range(int(secs * RATE))]

    def send_frames(samples):
        t_next = time.time()
        for i in range(0, len(samples) - FRAME + 1, FRAME):
            c = samples[i:i + FRAME]
            ws.send({"type": "input_audio_buffer.append", "audio": base64.b64encode(struct.pack(f"<{len(c)}h", *c)).decode()})
            sent_at.append(time.time() - t0)
            user.extend(c)
            t_next += FRAME / RATE
            time.sleep(max(0.0, t_next - time.time()))

    send_frames(quiet(a.lead))
    marks = []
    for i, path in enumerate(a.say):
        if a.barge_at is not None and i == len(a.say) - 1:
            # speak over it: wait until its text has been going barge_at seconds
            t_wait = time.time() + 30
            buf = []
            while time.time() < t_wait and not (st["first_text"] is not None and time.time() - t0 > st["first_text"] + a.barge_at):
                buf = quiet(FRAME / RATE)
                send_frames(buf)
        clip = read_wav(path)
        clip += [0] * ((-len(clip)) % FRAME)
        marks.append({"say": os.path.basename(path), "from_s": round(len(user) / RATE, 2), "to_s": round((len(user) + len(clip)) / RATE, 2)})
        send_frames(clip)
        gap = a.gap[i] if i < len(a.gap) else 5.0
        send_frames(quiet(gap))
    deadline = time.time() + 10
    while time.time() < deadline and len(st["deltas"]) < len(sent_at) + 1:
        time.sleep(0.05)
    ws.send({"type": "session.close"})
    time.sleep(0.3)

    model = list(struct.unpack(f"<{len(st['model']) // 2}h", bytes(st["model"])))
    write_wav(os.path.join(a.out, "model.wav"), model)
    write_wav(os.path.join(a.out, "user.wav"), user)
    # the model is one frame ahead: its frame k + 1 answers the user's frame k, so frame k + 1 plays beside user frame k + 1
    n = max(len(user), len(model))
    mix = []
    for i in range(n):
        mix.append(user[i] if i < len(user) else 0)
        mix.append(model[i] if i < len(model) else 0)
    write_wav(os.path.join(a.out, "call.wav"), mix, channels=2)
    with open(os.path.join(a.out, "text.jsonl"), "w") as f:
        for t, d in st["text"]:
            f.write(json.dumps({"t": t, "delta": d}, ensure_ascii=False) + "\n")
    # turnaround: delta k + 1 (k from 0) answers the frame sent at sent_at[k]
    turns = [st["deltas"][k + 1][0] - sent_at[k] for k in range(min(len(sent_at), len(st["deltas"]) - 1))]
    turns.sort()
    text = "".join(d for _, d in st["text"])
    summary = {"frames_sent": len(sent_at), "frames_back": len(st["deltas"]), "model_audio_s": round(len(model) / RATE, 2),
               "turnaround_ms": {"median": round(1e3 * turns[len(turns) // 2], 1), "p95": round(1e3 * turns[len(turns) * 95 // 100], 1),
                                 "worst": round(1e3 * turns[-1], 1)} if turns else None,
               "first_frame_after_connect_s": round(st["deltas"][0][0], 2) if st["deltas"] else None,
               "user_said": marks, "speech_started": [round(x, 2) for x in st["started"]], "speech_stopped": [round(x, 2) for x in st["stopped"]],
               "model_text": " ".join(text.split()), "lockstep": (st["session"] or {}).get("nextsycl", {}).get("lockstep"), "errors": st["errors"]}
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
