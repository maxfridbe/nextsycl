/** Page shell (H3's layout): the title and the GPU pill, the metric strip, then the panels in the order they are
 *  used - what to make, its progress, the pictures it made, the session's pictures. */
import { jsx } from "../jsx.js";
import { state } from "../state.js";
import { Create } from "./Create.js";
import { Gallery, Results, Zoom } from "./Gallery.js";
import { Panel } from "./Panel.js";
import { Progress } from "./Progress.js";
import { Stats } from "./Stats.js";

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
    g.temp_pkg != null ? chip("temp", `${Math.round(g.temp_pkg)}°C` + (g.temp_vram != null ? ` · mem ${Math.round(g.temp_vram)}°C` : "")) : null,
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
  const title = info ? `${info.model}` : "nextsycl image";
  return (
    <div id="app">
      <div class="top">
        <h1><span class="i" props={{ innerHTML: "&#xf03e;" }} /> {title}</h1>
        <GpuPill />
      </div>
      <Stats />
      {state.error && !state.busy ? <pre class="err">{state.error}</pre> : null}
      <Panel id="create" icon="&#xf040;" title="Create" hint={info?.arch ?? (info?.loading ? "loading the model…" : "")}>
        <Create />
      </Panel>
      <Progress />
      {state.results.length
        ? <Panel id="result" icon="&#xf03e;" title="Result" hint={`${state.results.length} picture${state.results.length > 1 ? "s" : ""}`}>
            <Results />
          </Panel>
        : null}
      <Panel id="gallery" icon="&#xf1da;" title="Pictures" hint={`${state.history.length} made here, newest first`}>
        <Gallery />
      </Panel>
      <Zoom />
    </div>
  );
}
