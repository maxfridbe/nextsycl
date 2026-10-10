/** Boot, the state, the actions: poll /api/state, keep the tree in sync (H3's scheme). */
import { init } from "../../vendor/snabbdom/init.js";
import { attributesModule } from "../../vendor/snabbdom/modules/attributes.js";
import { classModule } from "../../vendor/snabbdom/modules/class.js";
import { eventListenersModule } from "../../vendor/snabbdom/modules/eventlisteners.js";
import { propsModule } from "../../vendor/snabbdom/modules/props.js";
import { styleModule } from "../../vendor/snabbdom/modules/style.js";
import type { VNode } from "../../vendor/snabbdom/vnode.js";
import { App } from "./components/App.js";
import type { Spec, State } from "./types.js";

export const ui = {
  state: null as State | null,
  spec: null as Spec | null,
  error: null as string | null,
  tab: (() => { try { return localStorage.getItem("ns-tab") ?? "status"; } catch { return "status"; } })() as "status" | "api",
  /** per model: the GPUs picked before enabling */
  pick: {} as Record<string, number[]>,
  /** per model: an action under way from this page (its button spins until the state says it is done) */
  pending: {} as Record<string, { what: string; since: number }>,
  /** the run whose output is open */
  openRun: null as number | null,
  copied: null as string | null,
};

const patch = init([attributesModule, propsModule, classModule, styleModule, eventListenersModule]);
let tree: VNode | Element = document.getElementById("app") as Element;
let scheduled = false;

export function render(): void {
  if (scheduled) return;
  scheduled = true;
  requestAnimationFrame(() => {
    scheduled = false;
    tree = patch(tree, App() as VNode);
  });
}

/** An action is done when the state shows its result (or after 20 min) */
function settle(): void {
  const st = ui.state;
  if (!st) return;
  for (const [id, p] of Object.entries(ui.pending)) {
    const m = st.models.find((x) => x.id === id);
    const done = !m || Date.now() - p.since > 1_200_000
      || (p.what === "load" && ["loaded", "busy", "ready"].includes(m.state))
      || (p.what === "unload" && !["loaded", "busy", "loading"].includes(m.state))
      || (p.what === "enable" && m.enabled) || (p.what === "disable" && !m.enabled);
    // a failed load shows in the actions: stop spinning
    const failed = p.what === "load" && m && m.state === "unloaded" && Date.now() - p.since > 8000
      && st.runs.some((r) => r.label === `load ${id}` && r.started * 1000 > p.since && r.rc !== null && r.rc !== 0);
    if (done || failed) delete ui.pending[id];
  }
}

async function poll(): Promise<void> {
  try {
    const r = await fetch("/api/state");
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    ui.state = (await r.json()) as State;
    ui.error = null;
    settle();
  } catch (e) {
    ui.error = `nextsycl serve does not answer (${String(e)})`;
  }
  render();
  const busy = Object.keys(ui.pending).length > 0 || ui.state?.runs.some((x) => x.rc === null);
  setTimeout(() => void poll(), busy ? 1000 : 2500);
}

export async function loadSpec(): Promise<void> {
  try {
    const r = await fetch("/api/spec");
    ui.spec = (await r.json()) as Spec;
  } catch (e) {
    ui.error = String(e);
  }
  render();
}

/** enable / disable / load / unload a model; its button spins until the state shows the result */
export async function act(what: string, model: string, extra: Record<string, unknown> = {}): Promise<void> {
  ui.pending[model] = { what, since: Date.now() };
  render();
  try {
    const r = await fetch("/api/action", {
      method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ do: what, model, ...extra }),
    });
    const j = (await r.json()) as { error?: string };
    if (!r.ok) {
      ui.error = j.error ?? `HTTP ${r.status}`;
      delete ui.pending[model];
    }
    if (what === "enable" || what === "disable") void loadSpec();
  } catch (e) {
    ui.error = String(e);
    delete ui.pending[model];
  }
  render();
  void poll();
}

export function setTab(t: "status" | "api"): void {
  ui.tab = t;
  try { localStorage.setItem("ns-tab", t); } catch { /* no storage */ }
  if (t === "api" && !ui.spec) void loadSpec();
  render();
}

export async function copy(text: string, key: string): Promise<void> {
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    const t = document.createElement("textarea");
    t.value = text;
    document.body.appendChild(t);
    t.select();
    document.execCommand("copy");
    t.remove();
  }
  ui.copied = key;
  render();
  setTimeout(() => { if (ui.copied === key) { ui.copied = null; render(); } }, 1500);
}

render();
void poll();
if (ui.tab === "api") void loadSpec();
