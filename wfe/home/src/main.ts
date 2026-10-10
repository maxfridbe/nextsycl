/** Boot, the state, the actions: poll /api/state, keep the tree in sync (H3's scheme). */
import { init } from "../../vendor/snabbdom/init.js";
import { attributesModule } from "../../vendor/snabbdom/modules/attributes.js";
import { classModule } from "../../vendor/snabbdom/modules/class.js";
import { eventListenersModule } from "../../vendor/snabbdom/modules/eventlisteners.js";
import { propsModule } from "../../vendor/snabbdom/modules/props.js";
import { styleModule } from "../../vendor/snabbdom/modules/style.js";
import type { VNode } from "../../vendor/snabbdom/vnode.js";
import { App } from "./components/App.js";
import type { State } from "./types.js";

export const ui = {
  state: null as State | null,
  error: null as string | null,
  /** per startable kind: the model and GPU picked */
  pick: {} as Record<string, { model: string; gpu: number }>,
  /** a refusal that can be overridden (a card in use): the kind it was for */
  conflict: null as { kind: string; message: string } | null,
  /** the run whose output is open */
  openRun: null as number | null,
  sending: false,
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

async function poll(): Promise<void> {
  try {
    const r = await fetch("/api/state");
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    ui.state = (await r.json()) as State;
    ui.error = null;
  } catch (e) {
    ui.error = `nextsycl serve does not answer (${String(e)})`;
  }
  render();
  const running = ui.state?.runs.some((x) => x.rc === null);
  setTimeout(() => void poll(), running ? 1000 : 2500);
}

/** start / stop a kind's server, or pick the chat model ("none": no chat, its card free) */
export async function act(body: Record<string, unknown>): Promise<void> {
  ui.sending = true;
  ui.conflict = null;
  render();
  try {
    const r = await fetch("/api/action", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
    const j = (await r.json()) as { run?: number; error?: string };
    if (r.status === 409 && typeof body.kind === "string") ui.conflict = { kind: body.kind, message: j.error ?? "in use" };
    else if (!r.ok) ui.error = j.error ?? `HTTP ${r.status}`;
    else if (j.run) ui.openRun = j.run;
  } catch (e) {
    ui.error = String(e);
  }
  ui.sending = false;
  render();
  void poll();
}

render();
void poll();
