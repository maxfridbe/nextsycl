/** Project and clip data: loading a timeline, selecting a clip, holding and clearing.
 *
 *  Kept out of the components so the router can drive the same actions a click does, and
 *  out of the router so nothing here needs to know about URLs. Components call navigate();
 *  the router calls these.
 */
import { api, rpc } from "./api.js";
import { render, state } from "./state.js";

export async function loadTimeline(project?: string): Promise<void> {
  const p = project ?? state.timelineProject;
  if (!p) {
    state.timelineProject = null;
    state.timelineClips = [];
    render();
    return;
  }
  state.timelineProject = p;
  try {
    const r = await api.timeline(p);
    state.timelineClips = r.clips.slice().sort((a, b) => a.n - b.n);
    state.error = null;
  } catch (e) {
    state.error = String(e);
    state.timelineClips = [];
  }
  render();
}

/** Follow whatever is rendering; otherwise stay put, else fall back to the newest project.
 *  Returns the project it settled on so the router can put it in the URL. */
export function autoSelectProject(): string | null {
  const all = state.status?.all_projects ?? [];
  if (!all.length) {
    state.timelineProject = null;
    return null;
  }
  const want = state.status?.job?.project ?? "";
  const before = state.timelineProject;
  if (want && all.includes(want)) state.timelineProject = want;
  else if (!state.timelineProject || !all.includes(state.timelineProject)) {
    state.timelineProject = all[0] ?? null;
  }
  if (state.timelineProject !== before) void loadTimeline();
  return state.timelineProject;
}

/** Open a clip: select it, fetch its record, cue the video if it has rendered. */
export async function selectClip(n: number | null, scroll = true): Promise<void> {
  state.selected = n;
  if (n === null) {
    state.detail = null;
    render();
    return;
  }
  const c = state.timelineClips.find((x) => x.n === n);
  render();
  if (!c) return;
  try {
    if (c.state === "done" && c.name) {
      state.video = { file: `${c.name}.mp4`, poster: c.thumb, playing: false };
      state.detail = { kind: "job", name: c.name, data: await api.job(c.name) };
    } else {
      state.detail = { kind: "clip", label: c.label, data: await api.clip(c.label) };
    }
    state.error = null;
  } catch (e) {
    state.error = String(e);
  }
  render();
  if (scroll) scrollToClip(n);
}

/** Bring a clip into view: the strip scrolls sideways, the page scrolls to the strip.
 *  Runs after the frame that drew it, so the element exists and has a width. */
export function scrollToClip(n: number): void {
  requestAnimationFrame(() => {
    const card = document.querySelector<HTMLElement>(`.tlc[data-n="${n}"]`);
    const strip = document.querySelector<HTMLElement>(".tl");
    if (!card || !strip) return;
    const target = card.offsetLeft - strip.clientWidth / 2 + card.offsetWidth / 2;
    strip.scrollTo({ left: Math.max(0, target), behavior: "smooth" });
    strip.scrollIntoView({ block: "nearest", behavior: "smooth" });
  });
}

export async function refreshStatus(): Promise<void> {
  state.status = await api.status();
  render();
}

export async function holdProject(project: string, on: boolean): Promise<void> {
  await api.hold({ project, on });
  await refreshStatus();
}

export async function clearProject(project: string): Promise<void> {
  if (!window.confirm(`Drop queued clips for ${project}?`)) return;
  await rpc("queue.clear", { project });
  await refreshStatus();
}
