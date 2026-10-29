/** Page shell (H3's layout): the title and the GPU pill, then the panels in the order they are used - what to make,
 *  its progress, the viewer, the meshes made here. */
import { jsx } from "../jsx.js";
import { state } from "../state.js";
import { Spinner } from "./Spinner.js";
import { Create } from "./Create.js";
import { Gallery } from "./Gallery.js";
import { Panel } from "./Panel.js";
import { Progress } from "./Progress.js";
import { Stage } from "./Stage.js";

function GpuPill() {
  const g = state.gpu;
  if (!g) return <div class="pill"><span class="dot off" /><span>GPU telemetry…</span></div>;
  const used = Math.round((g.vram_used_mb / Math.max(g.vram_total_mb, 1)) * 100);
  const busy = !!state.progress?.busy;
  const chip = (k: string, v: string) => <span class="gc"><span class="gk">{k}</span> <b>{v}</b></span>;
  const parts = [
    chip("GPU", g.name.replace("Intel(R) Arc(TM) ", "").replace(" Graphics", "")),
    g.power_w != null ? chip("power", `${Math.round(g.power_w)} W`) : null,
    chip("VRAM", `${(g.vram_used_mb / 1024).toFixed(1)}/${(g.vram_total_mb / 1024).toFixed(0)} GB (${used}%)`),
    g.temp_pkg != null ? chip("temp", `${Math.round(g.temp_pkg)}°C`) : null,
  ].filter((x) => x != null);
  return (
    <div class="pill gpu" attrs={{ title: g.name }}>
      <span class={{ dot: true, busy }} />
      {parts.flatMap((c, i) => (i ? [<span class="sep">|</span>, c] : [c]))}
    </div>
  );
}

export function App() {
  const info = state.info;
  const ab = info?.abilities;
  const what = ab ? [ab.generate ? (ab.texture ? "textured meshes" : "meshes") : "", ab.views > 1 ? `${ab.views} views` : "", ab.rig ? "rigging" : "", ab.animate ? "animation" : ""]
    .filter(Boolean).join(" · ") : "connecting…";
  return (
    <div id="app">
      <div class="top">
        <h1><span class="i" props={{ innerHTML: "&#xf1b2;" }} /> {info ? info.model : "nextsycl 3D"}</h1>
        <GpuPill />
      </div>
      {state.error && !state.busy ? <pre class="err">{state.error}</pre> : null}
      {!info ? <Spinner label="Connecting to the model…" /> : null}
      <Panel id="create" icon="&#xf040;" title="Create" hint={what}>
        <Create />
      </Panel>
      <Progress />
      <Panel id="view" icon="&#xf06e;" title="View" hint={state.shown?.url.split("/").pop() ?? ""}>
        <Stage />
      </Panel>
      <Panel id="made" icon="&#xf1da;" title="Made here" hint="newest first">
        <Gallery />
      </Panel>
    </div>
  );
}
