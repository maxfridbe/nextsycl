/** Boot: build the patch function, start the router, poll the server, keep the tree in sync. */
import { init } from "../../vendor/snabbdom/init.js";
import { attributesModule } from "../../vendor/snabbdom/modules/attributes.js";
import { classModule } from "../../vendor/snabbdom/modules/class.js";
import { datasetModule } from "../../vendor/snabbdom/modules/dataset.js";
import { eventListenersModule } from "../../vendor/snabbdom/modules/eventlisteners.js";
import { propsModule } from "../../vendor/snabbdom/modules/props.js";
import { styleModule } from "../../vendor/snabbdom/modules/style.js";
import type { VNode } from "../../vendor/snabbdom/vnode.js";
import { api } from "./api.js";
import { App } from "./components/App.js";
import { autoSelectProject, loadTimeline } from "./project.js";
import { startRouter, syncHash } from "./router.js";
import { onRender, render, restorePreferences, state } from "./state.js";

const patch = init([
  attributesModule, propsModule, classModule, styleModule, datasetModule, eventListenersModule,
]);

let tree: VNode | Element = document.getElementById("app") as Element;
let scheduled = false;

function paint(): void {
  if (scheduled) return;
  scheduled = true;
  requestAnimationFrame(() => {
    scheduled = false;
    tree = patch(tree, App() as VNode);
  });
}

onRender(paint);

async function pollStatus(): Promise<void> {
  try {
    state.status = await api.status();
    // only pick a project for the user when the URL did not name one
    if (!state.timelineProject) {
      if (autoSelectProject()) syncHash();
    }
    state.error = null;
  } catch (e) {
    state.error = String(e);
  }
  render();
  const busy = state.status && !state.status.idle;
  setTimeout(() => void pollStatus(), busy ? 3000 : 8000);
}

async function pollSlow(): Promise<void> {
  try {
    const [summary, gpu, list, films] = await Promise.all([
      api.summary(), api.gpu(), api.list(), api.films(),
    ]);
    state.summary = summary;
    state.gpu = gpu;
    state.list = list;
    state.films = films;
  } catch {
    /* transient */
  }
  render();
  setTimeout(() => void pollSlow(), 30000);
}

async function pollTimeline(): Promise<void> {
  if (state.tab === "jobs" && state.timelineProject) await loadTimeline();
  setTimeout(() => void pollTimeline(), 30000);
}

/** The canvas grid needs the VRAM model and the measured cells. It changes only when a new
 *  cell is measured, so once at boot and then every few minutes is plenty. */
async function pollPlan(): Promise<void> {
  try {
    state.plan = await api.plan();
    render();
  } catch {
    /* transient */
  }
  setTimeout(() => void pollPlan(), 300000);
}

restorePreferences();
startRouter();
paint();
void pollStatus();
void pollSlow();
void pollTimeline();
void pollPlan();
