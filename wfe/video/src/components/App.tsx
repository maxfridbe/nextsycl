/** Page shell: header, metrics, the two tabs.
 *
 *  Order on the jobs tab is the order you use it in: the cued video, the timeline (which
 *  carries the current clip's progress, the project's progress, the strip and the detail
 *  of whatever is selected), then the queue, then what has finished, then the films cut
 *  from it. */
import { jsx } from "../jsx.js";
import { state } from "../state.js";
import { navigate } from "../router.js";
import { Completed } from "./Completed.js";
import { Creation } from "./Creation.js";
import { Player } from "./Detail.js";
import { Films } from "./Films.js";
import { ClipProgress } from "./Progress.js";
import { Queue } from "./Queue.js";
import { Stats } from "./Stats.js";
import { Timeline } from "./Timeline.js";

function GpuPill() {
  const g = state.gpu;
  if (!g) return <div class="pill"><span class="dot off" /><span>GPU telemetry…</span></div>;
  const used = Math.round((g.vram_used_mb / Math.max(g.vram_total_mb, 1)) * 100);
  const p = g.pcie;
  const link = p && p.cur.gen ? `PCIe ${p.cur.gen}.0 x${p.cur.width}` : null;
  // the card can do more than the slot gives it (Gen5 x16 card in a Gen3 x8 slot): say so on hover
  const linkTip = p
    ? `link ${p.cur.gts} GT/s x${p.cur.width} · card max Gen${p.card_max.gen} x${p.card_max.width}` +
      ` · slot max Gen${p.slot_max.gen} x${p.slot_max.width}`
    : "";
  const tip = [
    g.name,
    g.pkg_power_w != null ? `package ${g.pkg_power_w} W` : "",
    g.power_cap_w != null ? `power cap ${g.power_cap_w} W` : "",
    g.temp_vram_max != null ? `VRAM hottest channel ${g.temp_vram_max}°C` : "",
    g.temp_pcie != null ? `PCIe ${g.temp_pcie}°C` : "",
    g.freq_mhz ? `${g.freq_mhz} MHz` : "",
    g.fan_rpm ? `fan ${g.fan_rpm} rpm` : "",
    linkTip,
    g.host_load1 != null ? `host load ${g.host_load1} on ${g.host_cpus ?? "?"} CPUs` : "",
  ].filter(Boolean).join("\n");
  const vramT = g.temp_vram_max ?? g.temp_vram;
  // label + value chips with separators: the bar wraps instead of running off the header
  const chip = (k: string, v: string) => <span class="gc"><span class="gk">{k}</span> <b>{v}</b></span>;
  const parts = [
    chip("GPU", `${Math.round(g.busy_pct)}%`),
    g.power_w != null ? chip("power", `${Math.round(g.power_w)} W`) : null,
    chip("VRAM", `${(g.vram_used_mb / 1024).toFixed(1)}/${(g.vram_total_mb / 1024).toFixed(0)} GB (${used}%)`),
    g.temp_pkg != null ? chip("temp", `${g.temp_pkg}°C` + (vramT != null ? ` · mem ${vramT}°C` : "")) : null,
    link ? chip("link", link) : null,
    g.host_load1 != null ? chip("host load", g.host_load1.toFixed(1)) : null,
  ].filter((x) => x != null);
  return (
    <div class="pill gpu" attrs={{ title: tip }}>
      <span class={{ dot: true, on: g.busy_pct > 1 }} />
      {parts.flatMap((c, i) => (i ? [<span class="sep">|</span>, c] : [c]))}
    </div>
  );
}

export function App() {
  const jobs = state.tab === "jobs";
  return (
    <div id="app">
      <div class="top">
        <h1><span class="i" props={{ innerHTML: "&#xf03d;" }} /> nextsycl video — MiniMax H3</h1>
        <GpuPill />
      </div>
      <Stats summary={state.summary} />
      <div class="tabs">
        <button
          type="button"
          class={{ tab: true, on: !jobs }}
          on={{ click: () => navigate({ tab: "create" }) }}
        >
          <span class="i" props={{ innerHTML: "&#xf040;" }} />Creation
        </button>
        <button
          type="button"
          class={{ tab: true, on: jobs }}
          on={{ click: () => navigate({ tab: "jobs" }) }}
        >
          <span class="i" props={{ innerHTML: "&#xf03a;" }} />Jobs
        </button>
      </div>
      {jobs
        ? <div class="jobs">
            {/* no project yet means no timeline to carry it, so the clip bar stands alone */}
            {state.timelineProject ? null : <ClipProgress />}
            <Player />
            <Timeline />
            <Queue />
            <Completed />
            <Films />
          </div>
        : <Creation />}
    </div>
  );
}
