/** What the GPU is doing on the request running now: the phase, its bar, seconds. */
import { jsx } from "../jsx.js";
import { state } from "../state.js";
import { fmtT } from "../util.js";

const MAKE = ["image", "structure", "shape", "texture", "decode", "mesh", "bake"];
const RIG = ["sample", "skeleton", "skin", "motion"];

export function Progress() {
  const p = state.progress;
  const running = state.busy || !!p?.busy;
  if (!running) {
    return (
      <div class="clipprog idle">
        <div class="stage"><span class="i dim" props={{ innerHTML: state.error ? "&#xf071;" : "&#xf00c;" }} /> {state.status}</div>
      </div>
    );
  }
  const phase = p?.busy ? p.phase ?? "" : "";
  const at = p?.at ?? 0;
  const of = Math.max(p?.of ?? 1, 1);
  const pct = Math.round(100 * Math.min(at / of, 1));
  const phases = state.mode === "make" ? MAKE : RIG;
  const idx = phases.indexOf(phase);
  const stage = p?.waiting ? `waiting: ${p.waiting} ahead` : phase ? `${phase}: ${at}/${of}` : "starting";
  return (
    <div class="clipprog">
      <div class="bar"><div class="fill" style={{ width: `${Math.max(pct, 2)}%` }} /></div>
      <div class="stage">
        <span class="i" props={{ innerHTML: "&#xf110;" }} /> {stage}{" "}
        <span class="hint">{`${pct}%${p?.busy ? ` · ${fmtT(p.seconds ?? 0)} elapsed` : ""}`}</span>
      </div>
      <div class="phase">{phases.map((x, i) => <span class={{ on: i === idx, done: idx > i }}>{x}</span>)}</div>
    </div>
  );
}
