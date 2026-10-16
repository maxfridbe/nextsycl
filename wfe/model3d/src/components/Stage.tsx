/** The viewer: a canvas the WebGL viewer draws into, made once and kept across repaints (snabbdom's hooks), the
 *  shown mesh loaded when it changes; its toggles over it. */
import { jsx } from "../jsx.js";
import { render, saveForm, state } from "../state.js";
import { Viewer } from "../viewer.js";
import { kilo } from "../util.js";
import type { VNode } from "../../../vendor/snabbdom/vnode.js";

let viewer: Viewer | null = null;

function sync(): void {
  if (!viewer) return;
  Object.assign(viewer, state.view);
  const url = state.shown?.url ?? "";
  if (url && viewer.url !== url) {
    state.stats = "loading…";
    viewer.load(url).then((l) => {
      state.stats = `${kilo(l.triangles)} triangles · ${kilo(l.vertices)} vertices${l.textured ? " · textured" : ""}` +
        `${l.joints ? ` · ${l.joints} bones` : ""}${l.frames ? ` · ${l.frames} frames at ${Math.round(l.fps)} fps` : ""}`;
      render();
    }).catch((e: unknown) => {
      if (String(e).includes("superseded")) return;
      state.stats = `cannot show it: ${String(e instanceof Error ? e.message : e)}`;
      render();
    });
  }
  viewer.draw();
}

function toggle(k: keyof typeof state.view, label: string, icon: string) {
  return (
    <button class={{ tbtn: true, on: state.view[k] }} attrs={{ title: label }}
            on={{ click: () => { state.view[k] = !state.view[k]; saveForm(); render(); } }}>
      <span class="i" props={{ innerHTML: icon }} />{label}
    </button>
  );
}

export function Stage() {
  const m = state.shown;
  return (
    <div class="stage3d">
      <canvas key="gl" hook={{
        insert: (v: VNode) => {
          try {
            viewer = new Viewer(v.elm as HTMLCanvasElement);
          } catch (e) {
            state.stats = String(e);
          }
          sync();
        },
        postpatch: () => sync(),
        destroy: () => { viewer?.dispose(); viewer = null; },
      }} />
      <div class="tools">
        {toggle("wire", "wire", "&#xf0e8;")}
        {toggle("flat", "flat", "&#xf1b2;")}
        {m?.rigged ? toggle("skeleton", "bones", "&#xf0c1;") : null}
        {m?.animated ? toggle("playing", "play", "&#xf04b;") : null}
        {m ? <a class="tbtn" attrs={{ href: m.url, download: m.url.split("/").pop() ?? "mesh.glb" }}><span class="i" props={{ innerHTML: "&#xf019;" }} />GLB</a> : null}
      </div>
      <div class="ov">{m ? state.stats : "nothing made yet: a picture of an object, then Make"} {m ? " · drag to turn, wheel to zoom, right-drag to pan, double-click to reset" : ""}</div>
    </div>
  );
}
