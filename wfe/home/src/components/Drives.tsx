/** The drives the models are on (device, link, free space, measured read speed) with the move under way, and each
 *  model row's place on them with a button to move its files to another drive (system.drives / system.move). */
import { jsx } from "../jsx.js";
import { driveAct, ui } from "../main.js";
import type { Drive, Model, Move } from "../types.js";
import { Spinner } from "./Spinner.js";

const GIB = 1024 ** 3;
const gb = (b: number) => (b / GIB).toFixed(b < 10 * GIB ? 1 : 0);
/** a drive by its mount's last folder (home, 2TBSSD), else its id */
const short = (d: Drive) => d.mount.split("/").filter((x) => x).pop() ?? d.id;
const day = (at: number) => new Date(at * 1000).toISOString().slice(0, 10);
const mins = (s: number) => (s >= 60 ? `${Math.floor(s / 60)}m${String(Math.round(s % 60)).padStart(2, "0")}s` : `${Math.round(s)}s`);

function MoveBar(m: Move) {
  const live = m.running;
  const what = live
    ? `${m.state === "copying" ? `copying ${m.file_index}/${m.files}` : m.state} · ${gb(m.done / 2)} of ${gb(m.bytes)} GiB · ${m.rate_mb_s} MB/s${m.eta_s != null ? ` · ${mins(m.eta_s)} left` : ""}`
    : m.state === "done" ? `moved in ${mins(m.seconds)}` : `${m.state}: ${m.error ?? ""}`;
  return (
    <div class={{ movebar: true, bad: m.state === "failed" }}>
      <div class="mvhead">
        <span><b>{m.model}</b> <span class="hint">{m.from.join(", ")} &#x2192;</span> <b>{m.to}</b> <span class={m.state === "done" ? "ok" : live ? "warn" : "bad"}>{what}</span></span>
        {live ? <button class="ghost small" attrs={{ title: "stop: what it copied is removed, the registry stays as it was" }}
          on={{ click: () => { if (window.confirm(`Stop moving ${m.model}? What it copied so far is removed.`)) void driveAct({ do: "move_cancel" }); } }}>cancel</button> : null}
      </div>
      {live ? <div class="meter"><i style={{ width: `${Math.min(100, m.pct)}%`, background: "#3b82f6" }} /></div> : null}
      {live && m.file ? <div class="hint mvfile">{m.file}</div> : null}
      {m.warning ? <div class="warn mvfile">{m.warning}</div> : null}
    </div>
  );
}

function DriveRow(d: Drive) {
  const size = Math.max(d.size, 1);
  const other = Math.max(0, d.size - d.free - d.models_bytes);
  const low = d.free < d.margin;
  return (
    <div class="drow">
      <span class={{ dot: true, busy: d.measuring }} />
      <div class="mname">
        <div class="name">{d.model || d.disk || d.id} <span class="hint">{d.id}</span></div>
        <div class="sub">{d.link ?? d.fstype}{d.link ? ` · ${d.fstype}` : ""} · {d.mount} · models in {d.folder}</div>
      </div>
      <div class="dspace" attrs={{ title: `${gb(d.models_bytes)} GiB of model files (${d.models.length} models), ${gb(other)} GiB else, ${gb(d.free)} GiB free of ${gb(d.size)}` }}>
        <div class="meter">
          <i style={{ width: `${(d.models_bytes / size) * 100}%`, background: "#06b6d4" }} />
          <i style={{ width: `${(other / size) * 100}%`, background: "#475569" }} />
        </div>
        <div class="kv"><span class={low ? "bad" : ""}><b>{gb(d.free)}</b> GiB free</span><span>of {gb(d.size)}</span><span>models <b>{gb(d.models_bytes)}</b></span></div>
      </div>
      <div class="acts">
        <span attrs={{ title: d.measured ? `${gb(d.measured.bytes)} GiB read past the page cache in ${d.measured.seconds} s (${d.measured.file})` : "not measured yet: the rate is its link's" }}>
          {d.measured ? <span><b>{(d.measured.mb_s / 1000).toFixed(1)} GB/s</b> <span class="hint">{day(d.measured.at)}</span></span>
            : <span class="hint">{d.rate_gb_s != null ? `link ${d.rate_gb_s} GB/s` : "not measured"}</span>}
        </span>
        <button class="ghost small" attrs={{ disabled: d.measuring || !!ui.state?.drives?.measuring, title: "read up to 4 GiB of its largest model files past the page cache (nothing written); not while a model loads from it" }}
          on={{ click: () => void driveAct({ do: "measure", drive: d.id }) }}>{d.measuring ? <span class="spin" /> : null}{d.measuring ? "measuring" : "measure"}</button>
      </div>
    </div>
  );
}

export function DrivesPanel() {
  const dv = ui.state?.drives;
  if (!ui.state) return <section class="panel"><div class="pbody"><Spinner label="Loading the drives…" /></div></section>;
  if (!dv || !dv.drives.length) return null;
  return (
    <section class="panel">
      <div class="phead"><span class="i" props={{ innerHTML: "&#xf0a0;" }} /><span class="ptitle">Drives</span>
        <span class="hint">where the models' files are · a model that streams its weights from the disk while it serves wants the fastest</span></div>
      <div class="pbody">
        {dv.move ? MoveBar(dv.move) : null}
        {dv.drives.map(DriveRow)}
      </div>
    </section>
  );
}

/** A model row's drives: where its files are, and a move button for each other drive (disabled with the reason) */
export function ModelDisk(m: Model) {
  const dv = ui.state?.drives;
  const dm = dv?.models[m.id];
  if (!dv || !dm) return null;
  const name = (id: string) => { const d = dv.drives.find((x) => x.id === id); return d ? short(d) : id; };
  const on = Object.entries(dm.drives).map(([id, b]) => `${name(id)} ${gb(b)} GiB`).join(" + ");
  return (
    <div class="disk">
      <span class="hint" attrs={{ title: `${dm.files} files${dm.missing ? `, ${dm.missing} missing` : ""}` }}>
        <span class="i dim" props={{ innerHTML: "&#xf0a0;" }} /> {on || "no files"}{dm.streams ? <span class="tag" attrs={{ title: "it reads weights from the disk while it serves: its drive's speed is its speed" }}>streams</span> : null}
      </span>
      {dm.moving && dv.move?.running ? <span class="warn"> moving · {dv.move.pct}%</span> : null}
      {Object.entries(dm.move).map(([id, mv]) => {
        const d = dv.drives.find((x) => x.id === id);
        if (!d) return null;
        const from = Object.keys(dm.drives).map((f) => dv.drives.find((x) => x.id === f)).filter((x): x is Drive => !!x);
        const slower = dm.streams && from.some((f) => (f.rate_gb_s ?? 0) > (d.rate_gb_s ?? 0));
        const title = mv.why ? `can't move to ${d.id} now: ${mv.why}`
          : `move its files to ${d.id} (${d.folder}): ${gb(mv.need ?? 0)} GiB to write; each file read back, then the registry repointed, then the old files removed (those another model names stay)`;
        return (
          <button class={{ ghost: true, small: true, slower }} attrs={{ disabled: !!mv.why, title }}
            on={{ click: () => {
              let q = `Move ${m.id}'s files (${gb(dm.bytes)} GiB) to ${d.id}, ${d.folder}?\n\nEach file is copied and read back, then the registry points at the copies and the old files are removed (files another model uses stay).`;
              if (slower) q = `${m.id} reads its weights from the disk while it serves: on ${d.id} (${d.rate_gb_s ?? "?"} GB/s, slower than where it is) it will serve slower.\n\n` + q;
              if (window.confirm(q)) void driveAct({ do: "move", model: m.id, drive: d.id });
            } }}>
            &#x2192; {short(d)}
          </button>
        );
      })}
    </div>
  );
}
