/** What the cards are doing, one row a clip being made (one clip per card): stage, bar, ETA, its own cancel.
 *
 *  This sits directly above the project bar in the timeline panel, so the two read as a
 *  pair: this clip, then the whole film. Idle after a failure, it says why the last clip
 *  failed - the queue holds itself after two failures in a row, and without the reason
 *  the page only said "idle". */
import { jsx } from "../jsx.js";
import { api, rpc } from "../api.js";
import { navigate } from "../router.js";
import { refresh, render, state } from "../state.js";
import type { ClipRow } from "../types.js";
import { jobLabelBeside, jobName, clipNumber, fmtT } from "../util.js";

/** The clips a cancel was asked for: their buttons read "cancelling…" until the engine stops them at the next step
 *  (seconds, mid-step), so the click shows at once */
const cancelling = new Set<string>();

/** The clips being made, one row each (a studio that says only one: that one) */
export function runningClips(): ClipRow[] {
  const s = state.status;
  if (!s || s.idle) return [];
  if (s.running_clips) return s.running_clips;
  return s.job
    ? [{ label: s.job.label, title: s.job.title, name: s.job.name, project: s.job.project, job: s.job, seconds: s.job.seconds, steps: s.job.steps,
         width: s.job.width, height: s.job.height, engine: s.job.engine, stage: s.stage, pct: s.pct, eta: s.eta, elapsed: s.elapsed, error: s.error }]
    : [];
}

export function isCancelling(name?: string): boolean {
  return !!name && cancelling.has(name) && runningClips().some((c) => c.name === name);
}

/** Stop this clip, then show the result now rather than at the next poll. */
export async function cancelRunning(name?: string): Promise<void> {
  if (name) cancelling.add(name);
  render();
  try {
    await rpc("cancel", name ? { name } : {});
    state.error = null;
  } catch (e) {
    state.error = String(e);
  }
  await refresh();
}

function lastFailure() {
  const s = state.status;
  // the last clip to end, if it failed: while another still runs, the top-level fields describe that one
  const last = s?.ended_clips?.[0];
  const f = last ? last.failure : s?.failure;
  const err = last ? last.error : s?.error;
  const job = last ? last.job : s?.job;
  if (!s || (!f && !err) || (!last && (!s.idle || s.stage !== "failed"))) return null;
  // asked for: said plainly, not as a failure
  if (f?.code === "cancelled") {
    return <div class="hint">{`Last clip cancelled${job && jobName(job) ? ` · ${jobName(job)}` : ""}.`}</div>;
  }
  const message = f?.message ?? (err ?? "").split("\n")[0] ?? "";
  const label = job && jobName(job) ? `${jobName(job)} · ` : "";
  const size = job?.width && job.height ? `${job.width}x${job.height}, ${job.seconds ?? "?"} s` : "";
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
  return (
    <div>
      {runningClips().map(clipRow)}
      {lastFailure()}
    </div>
  );
}

function clipRow(c: ClipRow) {
  const pct = c.pct ?? 0;
  const icon = c.error ? "&#xf071;" : c.done ? "&#xf00c;" : "&#xf110;";
  const label = jobName(c);
  const project = c.project ?? null;
  const n = clipNumber(c.label ?? "");
  const jump = project && n
    ? () => navigate({ tab: "jobs", project, clip: n })
    : null;
  return (
    <div class="clipprog" key={c.name ?? label}>
      <div class="bar"><div class="fill" style={{ width: `${pct}%` }} /></div>
      <div class="stage">
        <span class="i" props={{ innerHTML: icon }} />{" "}
        {c.gpu != null ? <span class="hint">{`GPU ${c.gpu} · `}</span> : null}
        {label
          ? <a
              class={{ cliplink: true, plain: !jump }}
              attrs={{ title: jump ? "show this clip in the timeline" : "" }}
              on={jump ? { click: jump } : {}}
            >{label}</a>
          : null}
        {jobLabelBeside(c) ? <span class="hint"> {jobLabelBeside(c)}</span> : null}
        {label ? " · " : ""}{c.stage ?? ""}{" "}
        <span class="hint">
          {`${pct}%${c.eta ? ` · ${fmtT(c.eta)} left` : ""}${c.elapsed ? ` · ${fmtT(c.elapsed)} elapsed` : ""}`}
        </span>
        <button
          type="button"
          class="tbtn"
          attrs={{ disabled: isCancelling(c.name) }}
          on={{ click: () => void cancelRunning(c.name) }}
        >{isCancelling(c.name) ? "cancelling…" : "cancel"}</button>
      </div>
      {c.waiting_for ? <div class="hint">{c.waiting_for}</div> : null}
      {c.error ? <pre class="err">{c.error}</pre> : null}
    </div>
  );
}
