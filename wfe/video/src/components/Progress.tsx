/** What the GPU is doing on the one clip it is rendering right now: stage, bar, ETA.
 *
 *  This sits directly above the project bar in the timeline panel, so the two read as a
 *  pair: this clip, then the whole film. Idle after a failure, it says why the last clip
 *  failed - the queue holds itself after two failures in a row, and without the reason
 *  the page only said "idle". */
import { jsx } from "../jsx.js";
import { api, rpc } from "../api.js";
import { navigate } from "../router.js";
import { refresh, render, state } from "../state.js";
import { clipNumber, fmtT } from "../util.js";

/** The clip a cancel was asked for: its buttons read "cancelling…" until the engine stops it at its next step
 *  (seconds, mid-step), so the click shows at once */
let cancelling: string | null = null;

export function isCancelling(): boolean {
  const s = state.status;
  return !!cancelling && !!s && !s.idle && s.job?.name === cancelling;
}

/** Stop the running clip, then show the result now rather than at the next poll. */
export async function cancelRunning(): Promise<void> {
  cancelling = state.status?.job?.name ?? null;
  render();
  try {
    await rpc("cancel", {});
    state.error = null;
  } catch (e) {
    state.error = String(e);
  }
  await refresh();
}

function lastFailure() {
  const s = state.status;
  const f = s?.failure;
  if (!s || !s.idle || s.stage !== "failed" || (!f && !s.error)) return null;
  // asked for: said plainly, not as a failure
  if (f?.code === "cancelled") {
    return <div class="hint">{`Last clip cancelled${s.job?.label ? ` · ${s.job.label}` : ""}.`}</div>;
  }
  const message = f?.message ?? (s.error ?? "").split("\n")[0] ?? "";
  const label = s.job?.label ? `${s.job.label} · ` : "";
  const size = s.job?.width && s.job.height ? `${s.job.width}x${s.job.height}, ${s.job.seconds ?? "?"} s` : "";
  return (
    <div class="lastfail">
      <div>
        <span class="i err" props={{ innerHTML: "&#xf071;" }} />{" "}
        <b>{`Last clip failed`}</b>
        <span class="hint">{` · ${label}${size}${f?.code ? ` · ${f.code.replace(/_/g, " ")}` : ""}`}</span>
      </div>
      <div class="err">{message}</div>
      {f?.hint ? <div class="hint">{f.hint}</div> : null}
      {s.paused
        ? <div class="hint">
            {"The queue held itself after the failures. "}
            <button
              type="button"
              class="tbtn"
              on={{ click: () => { void api.hold({ on: false }).then(() => refresh()); } }}
            >resume the queue</button>
          </div>
        : null}
    </div>
  );
}

export function ClipProgress() {
  const s = state.status;
  if (!s) return <div />;
  if (s.idle) {
    const heldAll = s.queue_items.length > 0 && s.queue_items.every((x) => x.held);
    return (
      <div class="clipprog idle">
        <div class="stage">{heldAll ? "idle — every queued clip is held" : "idle"}</div>
        {lastFailure()}
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
          attrs={{ disabled: isCancelling() }}
          on={{ click: () => void cancelRunning() }}
        >{isCancelling() ? "cancelling…" : "cancel"}</button>
      </div>
      {s.error ? <pre class="err">{s.error}</pre> : null}
    </div>
  );
}
