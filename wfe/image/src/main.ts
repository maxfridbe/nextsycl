/** Boot: build the patch function, poll the server, keep the tree in sync (H3's scheme). */
import { init } from "../../vendor/snabbdom/init.js";
import { attributesModule } from "../../vendor/snabbdom/modules/attributes.js";
import { classModule } from "../../vendor/snabbdom/modules/class.js";
import { datasetModule } from "../../vendor/snabbdom/modules/dataset.js";
import { eventListenersModule } from "../../vendor/snabbdom/modules/eventlisteners.js";
import { propsModule } from "../../vendor/snabbdom/modules/props.js";
import { styleModule } from "../../vendor/snabbdom/modules/style.js";
import type { VNode } from "../../vendor/snabbdom/vnode.js";
import { adoptDefaults, describePicture, loadDescribers, refreshHistory } from "./actions.js";
import { api } from "./api.js";
import { App } from "./components/App.js";
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

/** The model and its options; until it has loaded, again soon. */
async function pollInfo(): Promise<void> {
  try {
    state.info = await api.info();
    if (!state.info.loading) adoptDefaults();
    state.error = null;
  } catch (e) {
    state.error = `the server does not answer (${String(e)})`;
  }
  render();
  setTimeout(() => void pollInfo(), state.info && !state.info.loading ? 30000 : 1500);
}

/** The request running: often while one runs, rarely when idle. */
async function pollProgress(): Promise<void> {
  try {
    state.progress = await api.progress();
  } catch {
    /* transient */
  }
  render();
  setTimeout(() => void pollProgress(), state.busy || state.progress?.busy ? 400 : 3000);
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

/** A picture on the clipboard goes to the Describe panel (text pastes are left alone). */
document.addEventListener("paste", (e: ClipboardEvent) => {
  const f = Array.from(e.clipboardData?.files ?? []).find((x) => x.type.startsWith("image/"));
  if (f && typeof state.describe.list === "object") void describePicture(f, f.name || "pasted.png");
});

restorePreferences();
paint();
void loadDescribers();
void pollInfo();
void pollProgress();
void pollGpu();
void refreshHistory();
