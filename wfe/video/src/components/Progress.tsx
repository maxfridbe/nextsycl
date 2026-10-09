/** What the GPU is doing on the one clip it is rendering right now: stage, bar, ETA.
 *
 *  This sits directly above the project bar in the timeline panel, so the two read as a
 *  pair: this clip, then the whole film. */
import { jsx } from "../jsx.js";
import { rpc } from "../api.js";
import { navigate } from "../router.js";
import { render, state } from "../state.js";
import { clipNumber, fmtT } from "../util.js";

export function ClipProgress() {
  const s = state.status;
  if (!s) return <div />;
  if (s.idle) {
    const heldAll = s.queue_items.length > 0 && s.queue_items.every((x) => x.held);
    return (
      <div class="clipprog idle">
        <div class="stage">{heldAll ? "idle — every queued clip is held" : "idle"}</div>
      </div>
    );
  }
  const pct = s.pct ?? 0;
  const icon = s.error ? "&#xf071;" : s.done ? "&#xf00c;" : "&#xf110;";
  const label = s.job?.label ?? "";
  const project = s.job?.project ?? null;
  const n = clipNumber(label);
  const jump = project && n
    ? () => navigate({ tab: "jobs", project, clip: n })
    : null;
  return (
    <div class="clipprog">
      <div class="bar"><div class="fill" style={{ width: `${pct}%` }} /></div>
      <div class="stage">
        <span class="i" props={{ innerHTML: icon }} />{" "}
        {label
          ? <a
              class={{ cliplink: true, plain: !jump }}
              attrs={{ title: jump ? "show this clip in the timeline" : "" }}
              on={jump ? { click: jump } : {}}
            >{label}</a>
          : null}
        {label ? " · " : ""}{s.stage ?? ""}{" "}
        <span class="hint">
          {`${pct}%${s.eta ? ` · ${fmtT(s.eta)} left` : ""}${s.elapsed ? ` · ${fmtT(s.elapsed)} elapsed` : ""}`}
        </span>
        <button
          type="button"
          class="tbtn"
          on={{ click: () => { void rpc("cancel", {}).then(() => render()); } }}
        >cancel</button>
      </div>
      {s.error ? <pre class="err">{s.error}</pre> : null}
    </div>
  );
}
