/** The meshes made here, newest first: their pictures, a click shows one. */
import { jsx } from "../jsx.js";
import { show } from "../actions.js";
import { state } from "../state.js";
import { fmtT, kilo } from "../util.js";

export function Gallery() {
  if (!state.history.length) return <div class="hint">none yet</div>;
  return (
    <div class="grid3d">
      {state.history.map((m) => (
        <div class={{ card3d: true, on: state.shown?.url === m.url }} on={{ click: () => show(m) }} attrs={{ title: m.prompt ?? m.url }}>
          {m.picture ? <img attrs={{ src: m.picture, loading: "lazy", alt: "" }} /> : <div class="noimg"><span class="i" props={{ innerHTML: "&#xf1b2;" }} /></div>}
          <div class="meta">
            <b>{kilo(m.triangles)} tris</b>
            <span>{[m.textured ? "tex" : "", m.rigged ? `${m.rigged} bones` : "", m.animated ? `${m.animated} fr` : "", fmtT(m.took)].filter(Boolean).join(" · ")}</span>
          </div>
        </div>
      ))}
    </div>
  );
}
