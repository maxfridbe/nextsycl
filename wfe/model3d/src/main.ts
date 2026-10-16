/** Boot: build the patch function, poll the server, keep the tree in sync (H3's scheme). */
import { init } from "../../vendor/snabbdom/init.js";
import { attributesModule } from "../../vendor/snabbdom/modules/attributes.js";
import { classModule } from "../../vendor/snabbdom/modules/class.js";
import { datasetModule } from "../../vendor/snabbdom/modules/dataset.js";
import { eventListenersModule } from "../../vendor/snabbdom/modules/eventlisteners.js";
import { propsModule } from "../../vendor/snabbdom/modules/props.js";
import { styleModule } from "../../vendor/snabbdom/modules/style.js";
import type { VNode } from "../../vendor/snabbdom/vnode.js";
import { refreshHistory } from "./actions.js";
import { api } from "./api.js";
import { App } from "./components/App.js";
import { onRender, render, restorePreferences, state } from "./state.js";

const patch = init([attributesModule, propsModule, classModule, styleModule, datasetModule, eventListenersModule]);

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

async function pollInfo(): Promise<void> {
  try {
    const i = await api.info();
    if (i.defaults || !state.info) state.info = i;
    state.error = null;
  } catch (e) {
    state.error = `the server does not answer (${String(e)})`;
  }
  render();
  setTimeout(() => void pollInfo(), state.info?.defaults ? 30000 : 1500);
}

async function pollProgress(): Promise<void> {
  try {
    const was = !!state.progress?.busy;
    state.progress = await api.progress();
    // a request from elsewhere (the API) ended: its mesh is in the gallery
    if (was && !state.progress.busy && !state.busy) void refreshHistory();
  } catch {
    /* transient */
  }
  render();
  setTimeout(() => void pollProgress(), state.busy || state.progress?.busy ? 700 : 3000);
}

async function pollGpu(): Promise<void> {
  try {
    state.gpu = await api.gpu();
  } catch {
    /* transient */
  }
  render();
  setTimeout(() => void pollGpu(), 3000);
}

restorePreferences();
paint();
void pollInfo();
void pollProgress();
void pollGpu();
void refreshHistory();
