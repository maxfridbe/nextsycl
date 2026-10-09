/** Finished clips, filterable. Clicking one jumps to it in its project's timeline, which
 *  cues the video and opens the detail in the one place they live. */
import { jsx } from "../jsx.js";
import { api } from "../api.js";
import { render, state } from "../state.js";
import { navigate } from "../router.js";
import type { ListRow } from "../types.js";
import { clipNumber, fmtMS, labelPrefix } from "../util.js";
import { Panel } from "./Panel.js";

/** A finished clip knows its project once the server has been told about projects; older
 *  records only have a label, so fall back to the prefix the timeline groups by. */
function projectOf(row: ListRow): string {
  return row.project || labelPrefix(row.label ?? "");
}

/** Jump to the clip in its timeline. Without a project there is no strip to jump to, so
 *  open the record directly and cue the video. */
async function open(row: ListRow): Promise<void> {
  const project = projectOf(row);
  const n = clipNumber(row.label ?? "");
  if (project && n) {
    navigate({ tab: "jobs", project, clip: n });
    return;
  }
  const name = row.file.replace(/\.mp4$/, "");
  state.video = { file: row.file, poster: `/thumb/${name}.webp`, playing: false };
  state.detail = { kind: "job", name, data: await api.job(name) };
  render();
}

export function Completed() {
  const f = state.filter.toLowerCase();
  const rows = state.list.filter(
    (x) => !f || `${x.file} ${x.label ?? ""} ${x.project ?? ""}`.toLowerCase().includes(f),
  );
  const filter = (
    <input
      class="filter"
      attrs={{ placeholder: "filter…", value: state.filter }}
      on={{ input: (e: Event) => { state.filter = (e.target as HTMLInputElement).value; render(); } }}
    />
  );
  return (
    <Panel
      id="completed"
      icon="&#xf1da;"
      title="Completed"
      hint={`${rows.length}${f ? ` of ${state.list.length}` : ""} clips`}
      tools={[filter]}
    >
      <div class="box">
        <table class="grid">
          <thead>
            <tr>
              <th>clip</th><th>label</th><th>project</th><th class="num">s</th><th class="num">steps</th>
              <th class="num">GPU</th><th class="num">Wh</th>
              <th class="num" attrs={{ title: "average card power over the run: Wh / GPU time" }}>W</th>
              <th class="num">speech</th>
              <th>engine</th><th>size</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((x) => {
              const n = clipNumber(x.label ?? "");
              const project = projectOf(x);
              const selected = state.timelineProject === project && state.selected === n && n > 0;
              return (
                <tr class={{ click: true, sel: selected }} on={{ click: () => void open(x) }}>
                  <td>{x.file.replace(/\.mp4$/, "").replace(/^h3_/, "")}</td>
                  <td class="lab accent">{x.label ?? ""}</td>
                  <td class="hint">{project}</td>
                  <td class="num">{x.seconds ?? ""}</td>
                  <td class="num">{x.steps ?? ""}</td>
                  <td class="num">{x.total ? fmtMS(x.total) : ""}</td>
                  <td class="num">{x.energy_wh ?? ""}</td>
                  <td class="num">{x.energy_wh && x.total ? Math.round((x.energy_wh * 3600) / x.total) : ""}</td>
                  <td class={{ num: true, warn: (x.speech_pct ?? 100) < 45 }}>
                    {x.speech_pct != null ? `${x.speech_pct}%` : ""}
                  </td>
                  <td class="hint">{x.engine ?? "Q4_K_M"}</td>
                  <td class="hint">{x.size ?? ""}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}
