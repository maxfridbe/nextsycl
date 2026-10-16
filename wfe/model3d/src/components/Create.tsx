/** What to make: pictures of an object (one, or several views for a multi-view model) and the settings - or, for a
 *  rigging or animating model, the mesh and the motion. */
import { jsx } from "../jsx.js";
import { addPictures, cancel, chooseMesh, go } from "../actions.js";
import { render, saveForm, state } from "../state.js";
import { SIDES } from "../util.js";

function num(label: string, key: "seed" | "steps" | "cfg" | "resolution" | "faces" | "textureSize" | "frames", hint: number | undefined, step = 1) {
  const v = state.form[key];
  return (
    <label>{label}
      <input attrs={{ type: "number", step, placeholder: hint === undefined ? "" : String(hint) }} props={{ value: v === null ? "" : String(v) }}
             on={{ change: (e: Event) => { const t = (e.target as HTMLInputElement).value; state.form[key] = t === "" ? null : Number(t); saveForm(); } }} />
    </label>
  );
}

function Pictures() {
  const f = state.form;
  const multi = (state.info?.abilities?.views ?? 1) > 1;
  return (
    <div>
      <div class="lab">{multi ? `pictures of the object (up to ${state.info?.abilities?.views} views)` : "a picture of the object"}
        <span class="hint">a plain or transparent background works best</span></div>
      <div class="views">
        {f.pictures.map((p, i) => (
          <div class="view">
            <button class="tbtn x" on={{ click: () => { f.pictures.splice(i, 1); render(); } }}>✕</button>
            <img attrs={{ src: p.data, alt: p.name }} />
            {multi
              ? <select class="sel" on={{ change: (e: Event) => { p.side = (e.target as HTMLSelectElement).value; } }}>
                  {SIDES.map((s) => <option attrs={{ value: s }} props={{ selected: s === p.side }}>{s}</option>)}
                </select>
              : null}
          </div>
        ))}
      </div>
      <label class="drop" on={{
        dragover: (e: DragEvent) => { e.preventDefault(); (e.currentTarget as HTMLElement).classList.add("over"); },
        dragleave: (e: DragEvent) => (e.currentTarget as HTMLElement).classList.remove("over"),
        drop: (e: DragEvent) => { e.preventDefault(); (e.currentTarget as HTMLElement).classList.remove("over"); if (e.dataTransfer) void addPictures(e.dataTransfer.files); },
      }}>
        <input attrs={{ type: "file", accept: "image/png,image/jpeg,image/webp", multiple: multi }} style={{ display: "none" }}
               on={{ change: (e: Event) => { const fl = (e.target as HTMLInputElement).files; if (fl) void addPictures(fl); } }} />
        drop {multi ? "pictures" : "a picture"} here, or click to choose (PNG, JPEG, WebP)
      </label>
    </div>
  );
}

function Mesh() {
  const f = state.form;
  const ab = state.info?.abilities;
  return (
    <div>
      <div class="lab">the mesh <span class="hint">GLB or OBJ, or one made here</span></div>
      <div class="row">
        <input attrs={{ type: "file", accept: ".glb,.obj,model/gltf-binary" }}
               on={{ change: (e: Event) => { const fl = (e.target as HTMLInputElement).files; if (fl?.[0]) void chooseMesh(fl[0]); } }} />
        <select class="sel" on={{ change: (e: Event) => { f.file = (e.target as HTMLSelectElement).value; if (f.file) { f.mesh = ""; f.meshName = ""; } render(); } }}>
          <option attrs={{ value: "" }}>…or one made here</option>
          {state.history.filter((m) => !m.rigged && !m.animated).map((m) => {
            const name = m.url.split("/").pop() ?? "";
            return <option attrs={{ value: name }} props={{ selected: f.file === name }}>{name}</option>;
          })}
        </select>
        {f.meshName ? <span class="hint">{f.meshName}</span> : null}
      </div>
      {ab?.animate
        ? <div><div class="lab">the motion</div>
            <input attrs={{ type: "text", placeholder: "a dragon flapping its wings" }} props={{ value: f.prompt }}
                   on={{ input: (e: Event) => { f.prompt = (e.target as HTMLInputElement).value; saveForm(); } }} /></div>
        : null}
    </div>
  );
}

export function Create() {
  const ab = state.info?.abilities;
  const d = state.info?.defaults;
  const f = state.form;
  // the model decides what can be asked: a generator makes, a rigger rigs
  if (ab) state.mode = ab.generate ? (state.mode === "rig" && (ab.rig || ab.animate) ? "rig" : "make") : "rig";
  const make = state.mode === "make";
  const busy = state.busy || !!state.progress?.busy;
  return (
    <div class="create">
      {ab && ab.generate && (ab.rig || ab.animate)
        ? <div class="tabs">
            <span class={{ tab: true, on: make }} on={{ click: () => { state.mode = "make"; render(); } }}>make</span>
            <span class={{ tab: true, on: !make }} on={{ click: () => { state.mode = "rig"; render(); } }}>rig</span>
          </div>
        : null}
      {make ? <Pictures /> : <Mesh />}
      <div class="row">
        {make && ab?.texture
          ? <label class="inline"><input attrs={{ type: "checkbox" }} props={{ checked: f.texture }}
                                          on={{ change: (e: Event) => { f.texture = (e.target as HTMLInputElement).checked; saveForm(); } }} /> textured</label>
          : null}
        {make
          ? <label class="inline"><input attrs={{ type: "checkbox" }} props={{ checked: f.removeBackground }}
                                          on={{ change: (e: Event) => { f.removeBackground = (e.target as HTMLInputElement).checked; saveForm(); } }} /> cut out the background</label>
          : null}
        <span class="seed">{num("seed", "seed", undefined)}
          <button class="tbtn" attrs={{ title: "a new seed" }} on={{ click: () => { f.seed = Math.floor(Math.random() * 1e6); saveForm(); render(); } }}>🎲</button></span>
        {make ? num("triangles", "faces", d?.faces) : null}
        {!make && ab?.animate ? num("frames", "frames", d?.frames) : null}
      </div>
      <div class="row">
        {num("steps", "steps", d?.steps)}
        {num("guidance", "cfg", d?.cfg, 0.1)}
        {make ? num("resolution", "resolution", d?.resolution) : null}
        {make && ab?.texture ? num("texture", "textureSize", d?.texture_size) : null}
      </div>
      <div class="row">
        <button attrs={{ id: "go", disabled: busy || !ab }} on={{ click: () => void go() }}>
          <span class="i" props={{ innerHTML: make ? "&#xf1b2;" : "&#xf0c1;" }} />{make ? "Make" : ab?.rig ? "Rig" : "Animate"}
        </button>
        {busy ? <button attrs={{ id: "stop" }} on={{ click: () => void cancel() }}><span class="i" props={{ innerHTML: "&#xf04d;" }} />Stop</button> : null}
      </div>
    </div>
  );
}
