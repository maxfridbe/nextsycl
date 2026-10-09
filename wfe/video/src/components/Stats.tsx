/** The metric strip: worker state, queue ETA, generation ratio, THIS project's ratio,
 *  energy, disk. */
import { jsx } from "../jsx.js";
import { api } from "../api.js";
import { render, state } from "../state.js";
import type { Disk, ProjectStats, Summary } from "../types.js";
import { fmtMS, fmtT } from "../util.js";

/** Global hold. Never kills the clip on the GPU: it stops the scheduler picking up the next
 *  one, and the server takes the worker container down once the current clip lands. Three
 *  states, because "paused" while a clip is still rendering is not the same as stopped. */
function WorkerCard() {
  const s = state.status;
  if (!s) return <div class="stat" />;
  const paused = !!s.paused;
  const busy = !s.idle;
  const held = s.queue_items.filter((x) => x.held).length;
  const toggle = () => { void api.hold({ on: !paused }).then(() => render()); };

  const [title, sub] = paused
    ? busy
      ? ["Finishing current clip", "then the worker stops; no new work will start"]
      : ["Paused", `worker stopped${held ? ` \u00b7 ${held} clips waiting` : ""}`]
    : busy
      ? ["Running", `${s.queue_items.length} queued`]
      : ["Idle", s.queue_items.length ? `${s.queue_items.length} queued` : "nothing queued"];

  return (
    <div class={{ stat: true, worker: true, paused, finishing: paused && busy }}>
      <div class="k">
        <span class="i dim" props={{ innerHTML: paused ? "&#xf04c;" : "&#xf04b;" }} /> worker
      </div>
      <div class="v">{title}</div>
      <div class="s">{sub}</div>
      <button
        type="button"
        class={{ wbtn: true, on: paused }}
        attrs={{ title: paused ? "resume picking up work" : "finish this clip, then stop" }}
        on={{ click: toggle }}
      >{paused ? "Resume" : "Pause after this clip"}</button>
    </div>
  );
}

const card = (icon: string, k: string, v: string, sub: string) => (
  <div class="stat">
    <div class="k">
      <span class="i dim" props={{ innerHTML: icon }} /> {k}
    </div>
    <div class="v">{v}</div>
    <div class="s">{sub}</div>
  </div>
);

const diskCard = (d: Disk) => {
  const colour = d.pct >= 90 ? "#ef4444" : d.pct >= 75 ? "#f59e0b" : "#22c55e";
  return (
    <div class="stat">
      <div class="k">
        <span class="i dim" props={{ innerHTML: "&#xf0a0;" }} /> disk {d.path}
      </div>
      <div class="v">{`${d.free_gb} GB free`}</div>
      <div class="bar" style={{ height: "8px", margin: "5px 0 3px" }}>
        <div class="fill" style={{ width: `${d.pct}%`, background: colour }} />
      </div>
      <div class="s">{`${d.used_gb} of ${d.total_gb} GB used · ${d.pct}%`}</div>
    </div>
  );
};

const projectCard = (p: ProjectStats) =>
  card(
    "&#xf0e4;",
    "project ratio",
    `${p.ratio}:1`,
    `${p.project} · ${p.clips} clips · ${fmtMS(p.video_s)} · ${fmtT(p.gpu_s)} GPU` +
      (p.wh ? ` · ${(p.wh / 1000).toFixed(2)} kWh` : ""),
  );

export function Stats(props: { summary: Summary | null }) {
  const s = props.summary;
  // the worker card must show even before the first summary lands, since pausing is the one
  // control you want reachable when things are going wrong
  if (!s) return <div class="stats"><WorkerCard /></div>;
  const eta = s.eta_ts ? new Date(s.eta_ts * 1000) : null;
  const etaText = eta
    ? `${eta.toLocaleDateString(undefined, { weekday: "short" })} ${eta.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" })}`
    : "idle";
  const ratio = s.rate_wall_per_video_s ? `${Math.round(s.rate_wall_per_video_s)}:1` : "—";
  return (
    <div class="stats">
      <WorkerCard />
      {card("&#xf252;", "queue ETA", s.running || s.queued ? etaText : "idle",
        s.running || s.queued ? `${fmtT(s.remaining_wall_s)} of GPU left` : "nothing queued")}
      {card("&#xf03a;", "queued", `${s.queued} clips`, `${fmtMS(s.queued_video_s)} of video`)}
      {card("&#xf0e4;", "generation ratio", `${ratio} (GPU : video)`,
        s.rate_wall_per_video_s ? `${fmtMS(s.rate_wall_per_video_s * 60)} per 1 min of film` : "")}
      {s.project_stats && s.project_stats.ratio ? projectCard(s.project_stats) : null}
      {card("&#xf0e7;", "energy", s.remaining_kwh != null ? `${s.remaining_kwh} kWh left` : "—",
        s.rate_wh_per_video_s ? `${s.rate_wh_per_video_s} Wh per second of film` : "")}
      {s.disk ? diskCard(s.disk) : null}
      {card("&#xf00c;", "completed", `${s.done_total} clips`,
        `${fmtMS(s.done_video_s)} of video · ${fmtT(s.done_wall_s)} GPU · ${(s.done_wh / 1000).toFixed(1)} kWh`)}
    </div>
  );
}
