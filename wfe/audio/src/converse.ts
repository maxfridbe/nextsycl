/** A live conversation with a full-duplex speech model (Sophon-Realtime; Haru-4B in lockstep): the microphone streamed
 *  to the server's /v1/realtime WebSocket (OpenAI Realtime's events, docs/api/realtime.md), its spoken answers played as
 *  they come, the running transcript. Opened through the front door (/open/<id>/) the socket goes to
 *  /open/<id>/v1/realtime, which the door passes on; on the model's own server to /v1/realtime.
 *
 *  A lockstep model (info.realtime.frame_ms > 0: Haru-4B) answers every 133 ms frame of the microphone with one of its
 *  own, silence included, in one long response: nothing is stopped when the user speaks (the model hears it and stops
 *  itself), its audio plays behind a short jitter buffer, and its text is split into lines where the user spoke. Its
 *  character is a card (nextsycl.card). */
import { render, state } from "./state.js";

export interface Line {
  role: "user" | "assistant" | "tool" | "note";
  text: string;
  id: string;
  /** an answer: in progress, done, cut short by the user */
  status?: "live" | "done" | "cut";
  /** seconds of the user's speech, or of the answer's audio */
  seconds?: number;
}

export interface Talk {
  on: boolean;
  connecting: boolean;
  /** listening | you are speaking | thinking | speaking */
  phase: string;
  lines: Line[];
  /** the microphone's level, 0..1 */
  level: number;
  error: string | null;
  /** first audio after the user stopped (s), the last answer's */
  latency: number | null;
  voice: string;
  language: string;
  instructions: string;
  /** a lockstep model's character card (its fields: persona, name, gender, backchannel, other_name, relationship,
   *  situation; empty ones keep the model's own) */
  card: Record<string, string>;
}

export const talk: Talk = { on: false, connecting: false, phase: "", lines: [], level: 0, error: null, latency: null, voice: "", language: "en", instructions: "",
                             card: {} };

let ws: WebSocket | null = null;
let ctx: AudioContext | null = null;
let mic: MediaStream | null = null;
let node: AudioWorkletNode | null = null;
let pending: Float32Array[] = [];
let pendingLen = 0;
/** the answer playing: its sources, when it started, the samples scheduled */
let sources: AudioBufferSourceNode[] = [];
let playhead = 0;
let playing: { id: string; start: number; samples: number } | null = null;
let stoppedAt = 0;
const OUT_RATE = 24000;
/** a lockstep model's audio plays this far behind (s): its frames come every 133 ms, give or take the network */
const JITTER = 0.2;
/** lockstep: the user spoke since the model's last text (its next text starts a new line), and the lines made */
let splitNext = false;
let part = 0;

/** whether the model runs in lockstep (a frame in, a frame out) */
export function lockstep(): boolean {
  return (state.info?.realtime?.frame_ms ?? 0) > 0;
}

const CAPTURE = `class Cap extends AudioWorkletProcessor {
  process(inputs) { const c = inputs[0] && inputs[0][0]; if (c) this.port.postMessage(c.slice(0)); return true; }
}
registerProcessor("ns-capture", Cap);`;

function b64(bytes: Uint8Array): string {
  let s = "";
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(s);
}

function unb64(s: string): Uint8Array {
  const bin = atob(s);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

function send(v: unknown): void {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(v));
}

function line(role: Line["role"], text: string, id: string, status?: Line["status"]): Line {
  const l: Line = { role, text, id, status };
  talk.lines.push(l);
  if (talk.lines.length > 200) talk.lines.splice(0, talk.lines.length - 200);
  return l;
}

function find(id: string): Line | undefined {
  for (let i = talk.lines.length - 1; i >= 0; i--) {
    const l = talk.lines[i];
    if (l && l.id === id) return l;
  }
  return undefined;
}

/** The microphone's frames: every ~80 ms one append (16-bit PCM at the context's rate) */
function onFrame(f: Float32Array): void {
  let e = 0;
  for (const x of f) e += x * x;
  const lvl = Math.min(1, Math.sqrt(e / Math.max(f.length, 1)) * 6);
  talk.level = 0.8 * talk.level + 0.2 * lvl;
  pending.push(f);
  pendingLen += f.length;
  if (!ctx || pendingLen < ctx.sampleRate * 0.08) return;
  const pcm = new Int16Array(pendingLen);
  let o = 0;
  for (const p of pending) {
    for (const x of p) pcm[o++] = Math.max(-32768, Math.min(32767, Math.round(x * 32767)));
  }
  pending = [];
  pendingLen = 0;
  send({ type: "input_audio_buffer.append", audio: b64(new Uint8Array(pcm.buffer)) });
}

/** An answer's audio (16-bit PCM at 24 kHz), queued after what is playing */
function play(id: string, delta: string): void {
  if (!ctx) return;
  const bytes = unb64(delta);
  const pcm = new Int16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength >> 1);
  if (pcm.length === 0) return;
  const buf = ctx.createBuffer(1, pcm.length, OUT_RATE);
  const ch = buf.getChannelData(0);
  pcm.forEach((x, i) => { ch[i] = x / 32768; });
  const src = ctx.createBufferSource();
  src.buffer = buf;
  src.connect(ctx.destination);
  // lockstep: a steady stream of frames, played behind a jitter buffer (again after an underrun)
  const at = lockstep() ? (playhead < ctx.currentTime + 0.01 ? ctx.currentTime + JITTER : playhead) : Math.max(ctx.currentTime + 0.04, playhead);
  src.start(at);
  src.onended = () => {
    sources = sources.filter((s) => s !== src);
    if (sources.length === 0 && talk.phase === "speaking") {
      talk.phase = "listening";
      render();
    }
  };
  sources.push(src);
  playhead = at + buf.duration;
  if (!playing || playing.id !== id) {
    playing = { id, start: at, samples: 0 };
    if (stoppedAt > 0) {
      talk.latency = Math.max(0, at - stoppedAt);
      stoppedAt = 0;
    }
  }
  playing.samples += pcm.length;
  if (!lockstep()) talk.phase = "speaking";
}

/** A lockstep model's text: a new line after the user spoke, else the running one */
function lockstepText(id: string, delta: string): void {
  // a silent frame after its words: it has stopped
  if (!delta.trim() && talk.phase === "it is speaking") talk.phase = "listening";
  let l = find(`${id}#${part}`);
  if (!l || splitNext) {
    if (!delta.trim()) return;
    if (l) l.status = "done";
    part += 1;
    splitNext = false;
    l = line("assistant", "", `${id}#${part}`, "live");
  }
  l.text += delta;
  if (delta.trim()) talk.phase = "it is speaking";
}

/** The user cut in: what plays stops, and the server hears how much of the answer was heard */
function bargeIn(): void {
  if (!ctx || !playing) return;
  const heard = Math.max(0, Math.min(ctx.currentTime - playing.start, playing.samples / OUT_RATE));
  const total = playing.samples / OUT_RATE;
  for (const s of sources) {
    try {
      s.stop();
    } catch {
      /* already ended */
    }
  }
  sources = [];
  playhead = 0;
  if (heard < total - 0.05) {
    send({ type: "conversation.item.truncate", item_id: `msg_${playing.id}`, content_index: 0, audio_end_ms: Math.round(heard * 1000) });
    const l = find(playing.id);
    if (l) l.status = "cut";
  }
  playing = null;
}

function onEvent(v: Record<string, unknown>): void {
  const t = String(v.type ?? "");
  const id = String((v.response_id as string) ?? ((v.response as Record<string, unknown>)?.id as string) ?? "");
  switch (t) {
    case "session.created":
    case "session.updated":
      talk.connecting = false;
      talk.on = true;
      if (!talk.phase) talk.phase = "listening";
      break;
    case "input_audio_buffer.speech_started":
      // lockstep: the model heard it too and stops itself if it wants to; nothing is cut here
      if (!lockstep()) bargeIn();
      talk.phase = "you are speaking";
      splitNext = true;
      line("user", "", String(v.item_id ?? `u${Date.now()}`), "live");
      break;
    case "input_audio_buffer.speech_stopped": {
      if (lockstep()) {
        const l = find(String(v.item_id ?? ""));
        if (l) l.status = "done";
        talk.phase = "listening";
        break;
      }
      talk.phase = "thinking";
      if (ctx) stoppedAt = ctx.currentTime;
      break;
    }
    case "input_audio_buffer.committed": {
      const l = find(String(v.item_id ?? ""));
      if (l) l.status = "done";
      break;
    }
    case "response.created":
      if (lockstep()) {
        talk.phase = "listening";
        splitNext = true;
      } else line("assistant", "", id, "live");
      break;
    case "response.audio_transcript.delta":
    case "response.output_audio_transcript.delta":
    case "response.text.delta":
    case "response.output_text.delta": {
      if (lockstep()) {
        lockstepText(id, String(v.delta ?? ""));
        break;
      }
      const l = find(id) ?? line("assistant", "", id, "live");
      l.text += String(v.delta ?? "");
      break;
    }
    case "response.audio.delta":
    case "response.output_audio.delta":
      play(id, String(v.delta ?? ""));
      break;
    case "response.function_call_arguments.done":
      line("tool", `${String(v.name)}(${String(v.arguments)})`, String(v.call_id ?? ""), "done");
      break;
    case "response.done": {
      const r = v.response as Record<string, unknown>;
      const l = find(lockstep() ? `${String(r?.id ?? "")}#${part}` : String(r?.id ?? ""));
      if (l) {
        l.status = r?.status === "completed" ? (l.status === "cut" ? "cut" : "done") : "cut";
        const ns = r?.nextsycl as Record<string, unknown> | undefined;
        if (ns?.audio_ms) l.seconds = Number(ns.audio_ms) / 1000;
      }
      if (talk.phase === "thinking") talk.phase = "listening";
      break;
    }
    case "error":
      talk.error = String((v.error as Record<string, unknown>)?.message ?? "error");
      break;
  }
  render();
}

/** The socket's address: beside the page (the door's /open/<id>/ or the server's own /) */
function url(): string {
  const u = new URL("v1/realtime", location.href);
  u.protocol = location.protocol === "https:" ? "wss:" : "ws:";
  return u.toString();
}

export async function start(): Promise<void> {
  if (talk.on || talk.connecting) return;
  talk.error = null;
  talk.connecting = true;
  render();
  try {
    mic = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 } });
    ctx = new AudioContext();
    const mod = URL.createObjectURL(new Blob([CAPTURE], { type: "text/javascript" }));
    await ctx.audioWorklet.addModule(mod);
    URL.revokeObjectURL(mod);
    const srcNode = ctx.createMediaStreamSource(mic);
    node = new AudioWorkletNode(ctx, "ns-capture");
    node.port.onmessage = (e: MessageEvent) => onFrame(e.data as Float32Array);
    srcNode.connect(node);
  } catch (e) {
    talk.connecting = false;
    talk.error = `the microphone: ${String(e)} (a page over plain http on another machine may not get one: open it as localhost or over https)`;
    stop();
    return;
  }
  ws = new WebSocket(url());
  ws.onopen = () => {
    const rate = ctx?.sampleRate ?? 48000;
    const session: Record<string, unknown> = {
      modalities: ["audio", "text"], input_audio_format: "pcm16", output_audio_format: "pcm16", turn_detection: { type: "server_vad" },
      nextsycl: { input_sample_rate: rate, language: talk.language },
    };
    if (talk.voice) session.voice = talk.voice;
    if (talk.instructions.trim()) session.instructions = talk.instructions.trim();
    if (lockstep()) {
      const card = Object.fromEntries(Object.entries(talk.card).map(([k, x]) => [k, x.trim()]).filter(([, x]) => x));
      if (Object.keys(card).length) (session.nextsycl as Record<string, unknown>).card = card;
    }
    send({ type: "session.update", session });
  };
  ws.onmessage = (e: MessageEvent) => {
    try {
      onEvent(JSON.parse(String(e.data)) as Record<string, unknown>);
    } catch {
      /* not JSON */
    }
  };
  ws.onerror = () => {
    talk.error = "the conversation's connection failed";
    render();
  };
  ws.onclose = () => {
    if (talk.on || talk.connecting) stop();
  };
}

export function stop(): void {
  splitNext = false;
  playhead = 0;
  talk.on = false;
  talk.connecting = false;
  talk.phase = "";
  talk.level = 0;
  for (const s of sources) {
    try {
      s.stop();
    } catch {
      /* ended */
    }
  }
  sources = [];
  playing = null;
  if (ws) {
    try {
      ws.close();
    } catch {
      /* closed */
    }
  }
  ws = null;
  node?.disconnect();
  node = null;
  mic?.getTracks().forEach((t) => t.stop());
  mic = null;
  void ctx?.close();
  ctx = null;
  render();
}

/** A saved voice or the model's own: what the session speaks in (applied at once while one runs) */
export function setVoice(v: string): void {
  talk.voice = v;
  if (talk.on) send({ type: "session.update", session: { voice: v } });
  render();
}

export function setLanguage(l: string): void {
  talk.language = l;
  if (talk.on) send({ type: "session.update", session: { nextsycl: { language: l } } });
  render();
}

/** A card field (before starting: the card is the prompt, a change while it runs starts the conversation over) */
export function setCard(field: string, value: string): void {
  talk.card[field] = value;
  render();
}

export function clearTranscript(): void {
  talk.lines = [];
  render();
}

/** whether the page can open one (the model converses) */
export function canTalk(): boolean {
  return !!state.info?.realtime;
}
