/** Page shell (H3's layout): the title and the GPU pill, the metric strip, then the panels in the order they are
 *  used - the pages and the instructions, the reading, the readings made here. */
import { jsx } from "../jsx.js";
import { state } from "../state.js";
import { History } from "./History.js";
import { Output } from "./Output.js";
import { Panel } from "./Panel.js";
import { Read } from "./Read.js";
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
  const r = state.result;
  return (
    <div id="app">
      <div class="top">
        <h1><span class="i" props={{ innerHTML: "&#xf15c;" }} /> {info ? info.model : "nextsycl ocr"}</h1>
        <GpuPill />
      </div>
      <Stats />
      {state.error && !state.busy ? <pre class="err">{state.error}</pre> : null}
      {info && !info.loading && info.reads === false
        ? <pre class="err">{`${info.model} makes pictures; this page reads them (serve an OCR model)`}</pre> : null}
      <Panel id="read" icon="&#xf0c5;" title="Pages" hint={info?.loading ? "loading the model…" : `${state.pages.length} page${state.pages.length === 1 ? "" : "s"}`}>
        <Read />
      </Panel>
      {state.busy || r
        ? <Panel id="out" icon="&#xf036;" title="Reading" hint={r ? `${r.tokens} tokens` : "reading…"}><Output /></Panel>
        : null}
      <Panel id="history" icon="&#xf1da;" title="Readings" hint={`${state.history.length} here, newest first`}>
        <History />
      </Panel>
    </div>
  );
}
