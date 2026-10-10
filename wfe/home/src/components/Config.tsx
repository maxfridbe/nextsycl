/** The Config tab: nextsycl.conf as settings (grouped, each with what it does, its default and what reads it) or as
 *  the file itself; a save copies the file first and lists what has to restart to see the change. */
import { jsx } from "../jsx.js";
import { render, restartService, saveConfig, ui } from "../main.js";
import type { SettingRow } from "../types.js";

const READER: Record<string, string> = {
  serve: "this page", switch: "the model switch", studio: "the video studio", video: "the video daemon", llm: "the chat server",
  image: "the image server", audio: "the music server", cli: "the command line (next run)",
};

function Row(s: SettingRow, extra = false) {
  const edited = s.name in ui.edits;
  const value = edited ? ui.edits[s.name] : s.file ?? "";
  return (
    <div class={{ srow: true, edited, env: !!s.env }}>
      <div class="sname"><code>{s.name}</code>{s.env ? <span class="tag busy" attrs={{ title: `the service's environment sets ${s.env}: it wins over the file` }}>environment</span> : null}</div>
      <input attrs={{ type: "text", value, placeholder: s.default ? `default: ${s.default}` : "", spellcheck: false }}
        props={{ value }}
        on={{ input: (e: Event) => { ui.edits[s.name] = (e.target as HTMLInputElement).value; render(); } }} />
      <div class="swhat">{s.what}{s.readers.length ? <span class="hint"> · read by {s.readers.map((r) => READER[r] ?? r).join(", ")}</span> : null}
        {extra ? <span class="hint"> · in the file</span> : null}</div>
    </div>
  );
}

export function ConfigView() {
  const c = ui.config;
  if (!c) return <div class="hint">loading the settings…</div>;
  const groups = [...new Set(c.settings.map((s) => s.group ?? ""))];
  const n = Object.keys(ui.edits).length;
  const saved = ui.saved;
  return (
    <div>
      <section class="panel">
        <div class="phead"><span class="i" props={{ innerHTML: "&#xf013;" }} /><span class="ptitle">Settings</span>
          <span class="hint">{c.path} · the environment wins over the file · each save keeps a copy (the last 10)</span></div>
        <div class="pbody">
          <div class="row">
            {ui.raw === null
              ? [<button attrs={{ disabled: !n }} on={{ click: () => void saveConfig(false) }}>save {n ? `${n} change${n > 1 ? "s" : ""}` : ""}</button>,
                 n ? <button class="ghost" on={{ click: () => { ui.edits = {}; render(); } }}>discard</button> : null,
                 <button class="ghost" on={{ click: () => { ui.raw = c.text; render(); } }}>edit the file</button>]
              : [<button on={{ click: () => void saveConfig(true) }}>save the file</button>,
                 <button class="ghost" on={{ click: () => { ui.raw = null; render(); } }}>back to the settings</button>]}
            <span class="hint">an empty value removes the setting (its default applies)</span>
          </div>
          {saved
            ? <div class="saved">
                {saved.changed.length
                  ? <span>saved: <b>{saved.changed.join(", ")}</b>. </span>
                  : <span>nothing changed. </span>}
                {saved.note ? <span class="warn">{saved.note}. </span> : null}
                {saved.restart.filter((r) => r !== "cli").length
                  ? [<span>to take effect, restart: </span>,
                     ...saved.restart.filter((r) => r !== "cli").map((r) => (
                       <button class="small" attrs={{ disabled: !!ui.restarting[r] }} on={{ click: () => void restartService(r) }}>
                         {ui.restarting[r] ? <span class="spin" /> : null}{READER[r] ?? r}
                       </button>
                     ))]
                  : null}
                {saved.restart.includes("cli") ? <span class="hint"> (the command line reads it on its next run)</span> : null}
              </div>
            : null}
          {ui.raw !== null
            ? <textarea class="conf" attrs={{ spellcheck: false }} props={{ value: ui.raw }}
                on={{ input: (e: Event) => { ui.raw = (e.target as HTMLTextAreaElement).value; } }} />
            : null}
        </div>
      </section>
      {ui.raw === null
        ? [
            ...groups.map((g) => (
              <section class="panel">
                <div class="phead"><span class="ptitle">{g}</span></div>
                <div class="pbody">{c.settings.filter((s) => (s.group ?? "") === g).map((s) => Row(s))}</div>
              </section>
            )),
            <section class="panel">
              <div class="phead"><span class="ptitle">Engine settings and others</span>
                <span class="hint">{c.prefixes.map((p) => `${p.prefix}*: ${p.what}`).join(" · ")}</span></div>
              <div class="pbody">
                {c.other.map((s) => Row(s, true))}
                <div class="srow">
                  <input attrs={{ type: "text", placeholder: "NS_QW_SOMETHING", value: ui.newName, spellcheck: false }} props={{ value: ui.newName }}
                    on={{ input: (e: Event) => { ui.newName = (e.target as HTMLInputElement).value.toUpperCase(); render(); } }} />
                  <input attrs={{ type: "text", placeholder: "value", value: ui.newValue }} props={{ value: ui.newValue }}
                    on={{ input: (e: Event) => { ui.newValue = (e.target as HTMLInputElement).value; } }} />
                  <button class="small" attrs={{ disabled: !/^[A-Z_][A-Z0-9_]*$/.test(ui.newName) }}
                    on={{ click: () => { ui.edits[ui.newName] = ui.newValue; ui.newName = ""; ui.newValue = ""; render(); } }}>add</button>
                </div>
                {Object.keys(ui.edits).filter((k) => !c.settings.some((s) => s.name === k) && !c.other.some((s) => s.name === k))
                  .map((k) => <div class="hint">new: <code>{k}={ui.edits[k]}</code> (save to keep)</div>)}
              </div>
            </section>,
          ]
        : null}
    </div>
  );
}
