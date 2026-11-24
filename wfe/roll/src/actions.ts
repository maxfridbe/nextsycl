/** What the buttons do: start, append, end, cancel (the generated client's video.* methods), and following the film. */
import type { VideoRollReq } from "../../home/src/gen/nextsycl.js";
import { api } from "./api.js";
import { render, say, state } from "./state.js";

async function act(what: string, f: () => Promise<unknown>): Promise<void> {
  state.busy = true;
  state.error = null;
  render();
  try {
    const r = await f();
    say(`${what}: ${JSON.stringify(r).slice(0, 240)}`);
  } catch (e) {
    state.error = `${what}: ${String(e)}`;
    say(state.error);
  }
  state.busy = false;
  render();
}

export function start(): Promise<void> {
  const events = state.events.filter((e) => e.prompt.trim()).map((e) => ({ prompt: e.prompt.trim(), seconds: e.seconds }));
  const req: VideoRollReq = { model: state.model, events, label: state.label, seed: state.seed, live: state.live, max_idle_s: state.maxIdle, queue: true };
  if (state.firstFrame.trim()) req.first_frame = state.firstFrame.trim();
  state.film = state.label;
  state.status = null;
  return act("video.roll", () => api["video.roll"](req));
}

export function append(): Promise<void> {
  if (!state.film || !state.add.prompt.trim()) return Promise.resolve();
  const ev = { prompt: state.add.prompt.trim(), seconds: state.add.seconds };
  state.add.prompt = "";
  return act("video.rollAppend", () => api["video.rollAppend"]({ label: state.film, events: [ev] }));
}

export function end(): Promise<void> {
  return state.film ? act("video.rollEnd", () => api["video.rollEnd"]({ label: state.film })) : Promise.resolve();
}

export function cancel(): Promise<void> {
  return state.film ? act("video.cancel", () => api["video.cancel"]({ label: state.film })) : Promise.resolve();
}

/** The film's status by its label, every 2 s while it runs (10 s once it ended) */
export async function follow(): Promise<void> {
  if (state.film) {
    try {
      state.status = (await api["video.status"]({ label: state.film })) as Record<string, any>;
    } catch (e) {
      if (!String(e).includes("not_found")) state.error = `video.status: ${String(e)}`;
    }
    render();
  }
  const ended = state.status?.idle === true;
  setTimeout(() => void follow(), ended ? 10000 : 2000);
}
