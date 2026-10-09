/** The queue, grouped by project, with holds and per-batch controls.
 *  Clicking a row takes you to that clip in the timeline rather than opening a panel of
 *  its own, so there is one place a selected clip is shown. */
import { jsx } from "../jsx.js";
import { api } from "../api.js";
import { render, state } from "../state.js";
import { clearProject, holdProject } from "../project.js";
import { navigate } from "../router.js";
import type { QueueItem } from "../types.js";
import { clipNumber, fmtMS } from "../util.js";
import { Panel } from "./Panel.js";

/** A queue row carries its position so the table can number it. */
type Row = QueueItem & { index: number };

const btn = (label: string, onclick: () => void) => (
  <button type="button" class="tbtn" on={{ click: onclick }}>{label}</button>
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

export function Queue() {
  const s = state.status;
  const items = s?.queue_items ?? [];
  if (!items.length) return <div />;
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
          {btn("timeline", () => navigate({ tab: "jobs", project, clip: null }))}
          {btn(paused ? "resume" : "pause", () => void holdProject(project, !paused))}
          {btn("clear", () => void clearProject(project))}
        </td>
      </tr>
    );
  };

  const clipRow = (project: string, x: Row) => {
    const n = clipNumber(x.label);
    const selected = state.timelineProject === project && state.selected === n && n > 0;
    return (
      <tr
        class={{ click: true, held: x.held, sel: selected }}
        on={{ click: () => navigate({ tab: "jobs", project, clip: n || null }) }}
      >
        <td class="num">{x.index + 1}</td>
        <td class="lab">{x.held ? "⏸ " : ""}{x.label}</td>
        <td class="num">{x.seconds}</td>
        <td class="num">{x.steps}</td>
        <td class="hint">{x.size}</td>
        <td class="hint">{x.engine ?? "Q4_K_M"}</td>
        <td>{x.chain ? <span class="i dim" props={{ innerHTML: "&#xf0c1;" }} /> : ""}</td>
      </tr>
    );
  };

  return (
    <Panel
      id="queue"
      icon="&#xf03a;"
      title="Queue"
      hint={`${items.length} clips · ${fmtMS(total)} of video${held ? ` · ${held} held` : ""}`}
      tools={[
        btn(s?.paused ? "resume all" : "pause all", () => {
          void api.hold({ on: !s?.paused }).then(() => render());
        }),
      ]}
    >
      <div class="box">
        <table class="grid">
          <thead>
            <tr>
              <th class="num">#</th><th>label</th><th class="num">s</th><th class="num">steps</th>
              <th>size</th><th>engine</th><th />
            </tr>
          </thead>
          <tbody>
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
