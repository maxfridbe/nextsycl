/** The queue, grouped by project, with holds and per-batch controls: the clip rendering now
 *  first (its progress and a cancel), then every queued clip with its own remove.
 *  Clicking a row takes you to that clip in the timeline rather than opening a panel of
 *  its own, so there is one place a selected clip is shown.
 *
 *  Every action refreshes the status at once (refresh()) - waiting for the next poll left a
 *  click looking ignored for up to 8 s - and a removed row goes from the table before the
 *  server answers, coming back only if the server refused. */
import { jsx } from "../jsx.js";
import { api, rpc } from "../api.js";
import { refresh, render, state } from "../state.js";
import { clearProject, holdProject } from "../project.js";
import { navigate } from "../router.js";
import type { QueueItem } from "../types.js";
import { clipNumber, fmtMS, fmtT } from "../util.js";
import { Panel } from "./Panel.js";
import { Spinner } from "./Spinner.js";
import { cancelRunning, isCancelling } from "./Progress.js";

/** A queue row carries its position so the table can number it and remove exactly it. */
type Row = QueueItem & { index: number };

const btn = (label: string, onclick: () => void, title = "") => (
  <button type="button" class="tbtn" attrs={{ title }} on={{ click: onclick }}>{label}</button>
);

/** A button inside a clickable row: its click is its own, not the row's. */
const rowBtn = (label: string, title: string, onclick: () => void, cls = "") => (
  <button
    type="button"
    class={cls ? { tbtn: true, rowx: true, [cls]: true } : { tbtn: true, rowx: true }}
    attrs={{ title, "aria-label": title }}
    on={{ click: (e: Event) => { e.stopPropagation(); onclick(); } }}
  >{label}</button>
);

function group(items: QueueItem[]): { order: string[]; rows: Map<string, Row[]> } {
  const order: string[] = [];
  const rows = new Map<string, Row[]>();
  items.forEach((item, index) => {
    const p = item.project || "(ungrouped)";
    if (!rows.has(p)) {
      rows.set(p, []);
      order.push(p);
    }
    rows.get(p)!.push({ ...item, index });
  });
  return { order, rows };
}

/** Drop one queued clip by its position (its label checked too when it has one). */
async function removeAt(x: Row): Promise<void> {
  const s = state.status;
  if (!s) return;
  const before = s.queue_items;
  s.queue_items = before.filter((_, i) => i !== x.index);
  render();
  try {
    await rpc("queue.remove", x.label ? { index: x.index, label: x.label } : { index: x.index });
    state.error = null;
  } catch (e) {
    state.error = `remove: ${String(e)}`;
  }
  await refresh();
}

async function act(p: Promise<unknown>): Promise<void> {
  try {
    await p;
    state.error = null;
  } catch (e) {
    state.error = String(e);
  }
  await refresh();
}

function runningRow() {
  const s = state.status;
  if (!s || s.idle || !s.job) return null;
  const j = s.job;
  const pct = s.pct ?? 0;
  return (
    <tr class="running">
      <td class="num"><span class="i" props={{ innerHTML: "&#xf110;" }} /></td>
      <td class="lab">
        {j.label || <span class="hint">(no label)</span>}
        <div class="minibar"><div class="fill" style={{ width: `${pct}%` }} /></div>
        <span class="hint">
          {`${s.stage ?? "running"} · ${pct}%${s.eta ? ` · ${fmtT(s.eta)} left` : ""}`}
        </span>
      </td>
      <td class="num">{j.seconds}</td>
      <td class="num">{j.steps}</td>
      <td class="hint">{`${j.width}x${j.height}`}</td>
      <td class="hint">{j.engine ?? ""}</td>
      <td class="acts">
        {isCancelling()
          ? <span class="hint">cancelling…</span>
          : rowBtn("cancel", "stop this clip (at its next step)", () => void cancelRunning(), "danger")}
      </td>
    </tr>
  );
}

export function Queue() {
  const s = state.status;
  const items = s?.queue_items ?? [];
  const running = runningRow();
  // held and empty still shows: its "resume all" is the way out of the hold the queue puts itself in after failures
  if (!s) return <Spinner label="Loading the queue…" />;
  if (!items.length && !running && !s?.paused) return <div />;
  const held = items.filter((x) => x.held).length;
  const total = items.reduce((a, x) => a + x.seconds, 0);
  const { order, rows } = group(items);
  const info = new Map((s?.projects ?? []).map((p) => [p.project, p]));

  const projectHeader = (project: string, mine: Row[]) => {
    const paused = !!info.get(project)?.paused;
    const seconds = mine.reduce((a, x) => a + x.seconds, 0);
    return (
      <tr class="ghead">
        <td attrs={{ colspan: 7 }}>
          <span class="i dim" props={{ innerHTML: "&#xf07b;" }} />{" "}
          <b class={{ heldname: paused }}>{project}</b>{" "}
          <span class="hint">{`${mine.length} clips · ${fmtMS(seconds)}${paused ? " · held" : ""}`}</span>
          {project === "(ungrouped)" ? null : btn("timeline", () => navigate({ tab: "jobs", project, clip: null }))}
          {project === "(ungrouped)" ? null : btn(paused ? "resume" : "pause", () => void holdProject(project, !paused))}
          {btn("clear", () => {
            if (project !== "(ungrouped)") return void clearProject(project);
            if (!window.confirm(`Drop the ${mine.length} ungrouped queued clips?`)) return;
            // the ungrouped ones one by one from the end, so the positions before them stay put
            void (async () => {
              for (const x of mine.slice().reverse()) await rpc("queue.remove", { index: x.index }).catch(() => undefined);
              await refresh();
            })();
          })}
        </td>
      </tr>
    );
  };

  const clipRow = (project: string, x: Row) => {
    const n = clipNumber(x.label);
    const selected = state.timelineProject === project && state.selected === n && n > 0;
    const go = project !== "(ungrouped)" && n > 0;
    return (
      <tr
        key={`${x.index}|${x.label}`}
        class={{ click: go, held: x.held, sel: selected }}
        on={go ? { click: () => navigate({ tab: "jobs", project, clip: n }) } : {}}
      >
        <td class="num">{x.index + 1}</td>
        <td class="lab">{x.held ? "⏸ " : ""}{x.label || <span class="hint">(no label)</span>}</td>
        <td class="num">{x.seconds}</td>
        <td class="num">{x.steps}</td>
        <td class="hint">{x.size}</td>
        <td class="hint">{x.engine ?? ""}</td>
        <td class="acts">
          {x.chain ? <span class="i dim" props={{ innerHTML: "&#xf0c1;" }} /> : ""}
          {rowBtn("✕", "remove this clip from the queue", () => void removeAt(x))}
        </td>
      </tr>
    );
  };

  return (
    <Panel
      id="queue"
      icon="&#xf03a;"
      title="Queue"
      hint={`${running ? "1 rendering · " : ""}${items.length} queued · ${fmtMS(total)} of video${held ? ` · ${held} held` : ""}`}
      tools={[
        btn(s?.paused ? "resume all" : "pause all", () => void act(api.hold({ on: !s?.paused }))),
      ]}
    >
      <div class="box">
        {state.error ? <div class="err">{state.error}</div> : null}
        <table class="grid">
          <thead>
            <tr>
              <th class="num">#</th><th>label</th><th class="num">s</th><th class="num">steps</th>
              <th>size</th><th>engine</th><th />
            </tr>
          </thead>
          <tbody>
            {running}
            {!items.length && !running
              ? <tr><td attrs={{ colspan: 7 }} class="hint">nothing queued · the queue is held: new clips wait until you resume it</td></tr>
              : null}
            {order.flatMap((project) => {
              const mine = rows.get(project) ?? [];
              return [projectHeader(project, mine), ...mine.map((x) => clipRow(project, x))];
            })}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}
