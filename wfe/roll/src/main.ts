/** Boot: build the patch function, follow the film, keep the tree in sync (H3's scheme). */
import { init } from "../../vendor/snabbdom/init.js";
import { attributesModule } from "../../vendor/snabbdom/modules/attributes.js";
import { classModule } from "../../vendor/snabbdom/modules/class.js";
import { eventListenersModule } from "../../vendor/snabbdom/modules/eventlisteners.js";
import { propsModule } from "../../vendor/snabbdom/modules/props.js";
import { styleModule } from "../../vendor/snabbdom/modules/style.js";
import type { VNode } from "../../vendor/snabbdom/vnode.js";
import { follow } from "./actions.js";
import { App } from "./components/App.js";
import { onRender, state } from "./state.js";

const patch = init([attributesModule, propsModule, classModule, styleModule, eventListenersModule]);

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
// ?label=NAME follows a film already started (by this page elsewhere, or video.roll)
const q = new URLSearchParams(location.search).get("label");
if (q) state.film = q;
paint();
void follow();
