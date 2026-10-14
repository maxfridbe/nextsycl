/** Boot: build the patch function, poll the server, keep the tree in sync (H3's scheme). */
import { init } from "../../vendor/snabbdom/init.js";
import { attributesModule } from "../../vendor/snabbdom/modules/attributes.js";
import { classModule } from "../../vendor/snabbdom/modules/class.js";
import { datasetModule } from "../../vendor/snabbdom/modules/dataset.js";
import { eventListenersModule } from "../../vendor/snabbdom/modules/eventlisteners.js";
import { propsModule } from "../../vendor/snabbdom/modules/props.js";
import { styleModule } from "../../vendor/snabbdom/modules/style.js";
import type { VNode } from "../../vendor/snabbdom/vnode.js";
import { addFiles, refreshHistory } from "./actions.js";
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
    state.info = await api.info();
    if (!state.error?.startsWith("the server")) render();
  } catch (e) {
    state.error = `the server does not answer (${String(e)})`;
  }
  render();
  setTimeout(() => void pollInfo(), state.info && !state.info.loading ? 30000 : 1500);
}

/** The reading running: its text so far, often while one runs */
async function pollProgress(): Promise<void> {
  try {
    state.progress = await api.progress();
  } catch {
    /* transient */
  }
  render();
  setTimeout(() => void pollProgress(), state.busy || state.progress?.busy ? 350 : 3000);
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

/** Pictures pasted anywhere on the page are pages too */
document.addEventListener("paste", (e) => {
  const files = Array.from(e.clipboardData?.files ?? ([] as File[]));
  if (files.length) void addFiles(files);
});

restorePreferences();
paint();
void pollInfo();
void pollProgress();
void pollGpu();
void refreshHistory();
