/** Joined films - the deliverables. Click loads one in the player without playing it. */
import { jsx } from "../jsx.js";
import { render, state } from "../state.js";
import { fmtMS } from "../util.js";
import { Panel } from "./Panel.js";

export function Films() {
  const films = state.films;
  return (
    <Panel
      id="films"
      icon="&#xf008;"
      title="Films"
      hint={`${films.length} joined · click to load at full resolution`}
    >
      <div class="box">
        <table class="grid">
          <thead>
            <tr><th>film</th><th class="num">length</th><th class="num">MB</th><th /></tr>
          </thead>
          <tbody>
            {films.length === 0
              ? <tr><td attrs={{ colspan: 4 }} class="hint">no joined films yet</td></tr>
              : films.map((f) => (
                <tr
                  class={{ click: true, sel: state.video?.file === f.file }}
                  on={{ click: () => { state.video = { file: f.file, poster: null, playing: false }; render(); } }}
                >
                  <td class="lab">{f.file}</td>
                  <td class="num">{f.seconds ? fmtMS(f.seconds) : ""}</td>
                  <td class="num">{f.mb}</td>
                  <td>
                    <a
                      attrs={{ href: `/out/${encodeURIComponent(f.file)}`, download: true, title: "download" }}
                      on={{ click: (e: Event) => e.stopPropagation() }}
                    >
                      <span class="i dim" props={{ innerHTML: "&#xf019;" }} />
                    </a>
                  </td>
                </tr>
              ))}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}
