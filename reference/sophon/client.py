"""A /v1/realtime client for checks (standard library only): streams a recording at real time - the user speaking -
then silence, and records what comes back: the events, the model's speech (answer.wav) and its text, and the timings
that matter for a conversation (the first audio after the user stopped, the real-time factor of the answer's audio).

    python3 -I client.py --url ws://127.0.0.1:8087/v1/realtime --wav q.wav --out DIR [--token T] [--voice chelsie]
                         [--lang en] [--silence 8] [--then q2.wav [--barge-at 1.0]] [--turns N]

--then: a second recording after the first answer (a second turn); --barge-at S: sent S seconds into the first
answer's audio instead (a barge-in: the answer should stop, the second turn be answered). Writes DIR/events.jsonl,
DIR/answer-N.wav (each answer's audio, 24 kHz) and DIR/summary.json.
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


class Ws:
    def __init__(self, url, token=None):
        u = urlparse(url)
        self.s = socket.create_connection((u.hostname, u.port or 80))
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
            rest = self.s.recv(65536)
            raise RuntimeError("no upgrade: " + (resp + rest).decode(errors="replace"))
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
    with wave.open(path, "rb") as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        assert sw == 2, "16-bit WAV expected"
        raw = w.readframes(w.getnframes())
    pcm = struct.unpack(f"<{len(raw) // 2}h", raw)
    if ch > 1:
        pcm = [sum(pcm[i:i + ch]) // ch for i in range(0, len(pcm), ch)]
    return list(pcm), sr


def write_wav(path, pcm16_bytes, rate=24000):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm16_bytes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--token")
    ap.add_argument("--wav", required=True)
    ap.add_argument("--then")
    ap.add_argument("--barge-at", type=float)
    ap.add_argument("--out", required=True)
    ap.add_argument("--voice")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--silence", type=float, default=8.0)
    ap.add_argument("--tools", action="store_true", help="offer a get_weather function and answer its calls")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ws = Ws(a.url, a.token)
    t0 = time.time()
    log = open(os.path.join(a.out, "events.jsonl"), "w")
    st = {"answers": {}, "order": [], "stopped": [], "first_audio": {}, "text": {}, "done": {}, "started": [], "calls": [], "errors": []}
    lock = threading.Lock()
    closed = threading.Event()

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
            slim = {k: (x if k != "delta" or "audio" not in t else f"<{len(x)} b64>") for k, x in v.items()}
            log.write(json.dumps({"t": round(now, 3), **slim}, ensure_ascii=False) + "\n")
            with lock:
                rid = v.get("response_id") or (v.get("response") or {}).get("id")
                if t == "input_audio_buffer.speech_started":
                    st["started"].append(now)
                elif t == "input_audio_buffer.speech_stopped":
                    st["stopped"].append(now)
                elif t == "response.created":
                    st["order"].append(rid)
                    st["answers"][rid] = bytearray()
                elif t in ("response.audio.delta", "response.output_audio.delta"):
                    st["answers"].setdefault(rid, bytearray()).extend(base64.b64decode(v["delta"]))
                    st["first_audio"].setdefault(rid, now)
                elif t in ("response.audio_transcript.delta", "response.output_audio_transcript.delta"):
                    st["text"][rid] = st["text"].get(rid, "") + v["delta"]
                elif t == "response.done":
                    st["done"][rid] = (now, v["response"]["status"])
                elif t == "response.function_call_arguments.done":
                    st["calls"].append(v)
                    out = json.dumps({"city": json.loads(v["arguments"]).get("city", "?"), "forecast": "sunny", "high_c": 21, "low_c": 12})
                    ws.send({"type": "conversation.item.create", "item": {"type": "function_call_output", "call_id": v["call_id"], "output": out}})
                    ws.send({"type": "response.create"})
                elif t == "error":
                    st["errors"].append(v["error"])
        closed.set()

    th = threading.Thread(target=reader, daemon=True)
    th.start()
    session = {"modalities": ["audio", "text"], "input_audio_format": "pcm16", "output_audio_format": "pcm16", "turn_detection": {"type": "server_vad"},
               "nextsycl": {"language": a.lang}}
    if a.voice:
        session["voice"] = a.voice
    if a.tools:
        session["tools"] = [{"type": "function", "name": "get_weather", "description": "The weather forecast for a city",
                             "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}]
    pcm, sr = read_wav(a.wav)
    session["nextsycl"]["input_sample_rate"] = sr
    ws.send({"type": "session.update", "session": session})
    step = sr * 80 // 1000

    def stream(samples, secs_silence=0.0, until=None):
        noise = [int(random.gauss(0, 8)) for _ in range(step)]
        chunks = [samples[i:i + step] for i in range(0, len(samples), step)] + [noise] * int(secs_silence / 0.08)
        t_next = time.time()
        for c in chunks:
            if until is not None and until():
                return
            ws.send({"type": "input_audio_buffer.append", "audio": base64.b64encode(struct.pack(f"<{len(c)}h", *c)).decode()})
            t_next += 0.08
            time.sleep(max(0.0, t_next - time.time()))

    t_user1 = time.time() - t0
    stream(pcm, 0.0)
    t_user1_end = time.time() - t0
    if a.then and a.barge_at is not None:
        # wait for the first answer's audio, then speak over it
        stream([], 30.0, until=lambda: bool(st["first_audio"]) and time.time() - t0 > min(st["first_audio"].values()) + a.barge_at)
        pcm2, sr2 = read_wav(a.then)
        stream(pcm2, a.silence)
    elif a.then:
        stream([], 30.0, until=lambda: len(st["done"]) >= 1)
        pcm2, sr2 = read_wav(a.then)
        stream(pcm2, a.silence)
    else:
        stream([], a.silence, until=lambda: bool(st["done"]) and not a.tools)
        # an answer still being made: wait for its end
        deadline = time.time() + 60
        while time.time() < deadline and len(st["done"]) < len(st["order"]):
            time.sleep(0.1)
        if a.tools:
            stream([], 8.0, until=lambda: len(st["done"]) >= 2)
    deadline = time.time() + 60
    while time.time() < deadline and len(st["done"]) < len(st["order"]):
        time.sleep(0.1)
    ws.send({"type": "session.close"})
    time.sleep(0.3)
    summary = {"user_speech": [round(t_user1, 2), round(t_user1_end, 2)], "speech_started": st["started"], "speech_stopped": st["stopped"],
               "answers": [], "calls": [{k: c[k] for k in ("name", "arguments")} for c in st["calls"]], "errors": st["errors"]}
    for i, rid in enumerate(st["order"]):
        b = bytes(st["answers"].get(rid, b""))
        write_wav(os.path.join(a.out, f"answer-{i + 1}.wav"), b)
        secs = len(b) / 2 / 24000
        fa = st["first_audio"].get(rid)
        done = st["done"].get(rid, (None, "?"))
        stop_before = [s for s in st["stopped"] if fa is not None and s <= fa]
        summary["answers"].append({"id": rid, "status": done[1], "text": st["text"].get(rid, ""), "audio_s": round(secs, 2),
                                   "first_audio_after_speech_end_s": round(fa - stop_before[-1], 3) if stop_before and fa else None,
                                   "first_audio_t": round(fa, 3) if fa else None, "done_t": round(done[0], 3) if done[0] else None})
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
