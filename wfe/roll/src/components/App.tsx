/** The page: the film to start (model, events, seed, live, max_idle_s), then the film as it is made - its status and
 *  ETA, the HLS stream playing as it grows, events added while it runs, end and cancel. */
import type { VNode } from "../../../vendor/snabbdom/vnode.js";
import { jsx } from "../jsx.js";
import { append, cancel, end, start } from "../actions.js";
import { fileUrl } from "../api.js";
import { attach } from "../player.js";
import { MODELS, render, state } from "../state.js";

const val = (e: Event) => (e.target as HTMLInputElement).value;
const secs = (s: unknown) => (typeof s === "number" && isFinite(s) ? `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, "0")}` : "-");

function Events() {
  return (
    <div>
      {state.events.map((ev, i) => (
        <div class="ev">
          <textarea attrs={{ placeholder: `event ${i + 1}: what happens` }} props={{ value: ev.prompt }} on={{ input: (e: Event) => { ev.prompt = val(e); } }} />
          <label>s <input attrs={{ type: "number", min: 1, max: 60, step: 1 }} props={{ value: String(ev.seconds) }}
                         on={{ input: (e: Event) => { ev.seconds = Number(val(e)) || 1; } }} /></label>
          <button class="alt" attrs={{ title: "remove this event", disabled: state.events.length < 2 }}
                  on={{ click: () => { state.events.splice(i, 1); render(); } }}>-</button>
        </div>
      ))}
      <button class="alt" on={{ click: () => { state.events.push({ prompt: "", seconds: 5 }); render(); } }}>+ event</button>
    </div>
  );
}

function Start() {
  const running = state.status && state.status.idle === false;
  return (
    <section class="panel">
      <h2>A film</h2>
      <div class="row">
        <label>model <select on={{ change: (e: Event) => { state.model = val(e); } }}>
          {MODELS.map((m) => <option attrs={{ value: m, selected: m === state.model }}>{m}</option>)}
        </select></label>
        <label>label <input attrs={{ type: "text" }} props={{ value: state.label }} on={{ input: (e: Event) => { state.label = val(e); } }} /></label>
        <label>seed <input attrs={{ type: "number" }} props={{ value: String(state.seed) }} on={{ input: (e: Event) => { state.seed = Number(val(e)) || 0; } }} /></label>
        <label><input attrs={{ type: "checkbox" }} props={{ checked: state.live }} on={{ change: (e: Event) => { state.live = (e.target as HTMLInputElement).checked; render(); } }} /> live</label>
        <label>max_idle_s <input attrs={{ type: "number", min: 0, step: 10 }} props={{ value: String(state.maxIdle) }}
                                 on={{ input: (e: Event) => { state.maxIdle = Number(val(e)) || 0; } }} /></label>
      </div>
      {state.model.includes("i2v") ? (
        <div class="row"><label>first_frame <input attrs={{ type: "text", placeholder: "a picture beside the clips (video.upload's name)" }}
          props={{ value: state.firstFrame }} on={{ input: (e: Event) => { state.firstFrame = val(e); } }} /></label></div>
      ) : null}
      <Events />
      <div class="row">
        <button attrs={{ disabled: state.busy || !!running }} on={{ click: () => void start() }}>Start the film</button>
        <span class="hint">video.roll: the events play one after another; live, it waits up to max_idle_s for more</span>
      </div>
    </section>
  );
}

function Film() {
  const s = state.status;
  if (!state.film) return null;
  const pct = typeof s?.pct === "number" ? s.pct : 0;
  const live = fileUrl(s?.live);
  const file = fileUrl(s?.file);
  const running = s && s.idle === false;
  return (
    <section class="panel">
      <h2>{state.film}</h2>
      <div class="stats">
        <div class="stat"><div class="k">stage</div><div class="v">{s?.stage ?? "asking"}</div><div class="s">{s?.waiting_for ?? ""}</div></div>
        <div class="stat"><div class="k">done</div><div class="v">{`${pct}%`}</div><div class="s">{s ? `${s.steps_done ?? 0} / ${s.steps_total ?? "?"} chunks` : ""}</div></div>
        <div class="stat"><div class="k">eta</div><div class="v">{running ? secs(s?.eta_remaining_s ?? s?.eta) : "-"}</div><div class="s">{`elapsed ${secs(s?.elapsed)}`}</div></div>
        <div class="stat"><div class="k">card</div><div class="v">{s?.gpu ?? "-"}</div><div class="s">{s?.failure?.code ?? (s?.done ? "made" : "")}</div></div>
      </div>
      <div class="bar"><div class="fill" style={{ width: `${pct}%` }} /></div>
      {s?.error ? <pre>{String(s.error)}</pre> : null}
      <video attrs={{ controls: true, muted: true, playsinline: true }} key="player"
             hook={{ insert: (v: VNode) => attach(v.elm as HTMLVideoElement, file ?? live, !file),
                     update: (_o: VNode, v: VNode) => attach(v.elm as HTMLVideoElement, file ?? live, !file) }} />
      <div class="hint">{file ? `the film: ${file}` : live ? `live: ${live} (HLS, growing)` : "the stream starts with its first segment"}</div>
      <div class="ev">
        <textarea attrs={{ placeholder: "the next event (video.rollAppend)" }} props={{ value: state.add.prompt }} on={{ input: (e: Event) => { state.add.prompt = val(e); } }} />
        <label>s <input attrs={{ type: "number", min: 1, max: 60 }} props={{ value: String(state.add.seconds) }}
                       on={{ input: (e: Event) => { state.add.seconds = Number(val(e)) || 1; } }} /></label>
        <button attrs={{ disabled: state.busy || !running }} on={{ click: () => void append() }}>Add</button>
      </div>
      <div class="row">
        <button class="alt" attrs={{ disabled: state.busy || !running }} on={{ click: () => void end() }}>End (video.rollEnd)</button>
        <button class="stop" attrs={{ disabled: state.busy || !running }} on={{ click: () => void cancel() }}>Cancel</button>
      </div>
      {s?.engine_log ? <div class="log">{(s.engine_log as string[]).join("\n")}</div> : null}
    </section>
  );
}

export function App() {
  return (
    <div id="app">
      <div class="top"><h1>MovieRoll</h1><span class="pill"><span class={{ dot: true, busy: state.busy }} />films by rolling diffusion, watched as they are made</span></div>
      <Start />
      {state.error ? <pre>{state.error}</pre> : null}
      <Film />
      {state.said.length ? <details attrs={{ open: true }}><summary>what the page asked</summary><div class="log">{state.said.join("\n")}</div></details> : null}
    </div>
  );
}
