/** What the GPU is doing on the request running now: the phase, its bar, seconds (H3's ClipProgress). The phases:
 *  the prompt, the tokens (the language model, frame by frame - the most of the time), the flow stage's steps, the
 *  decoder's windows. */
import { jsx } from "../jsx.js";
import { state } from "../state.js";
import { fmtT, mss } from "../util.js";

const PHASES = ["prompt", "tokens", "flow", "decode"];

export function Progress() {
  const p = state.progress;
  const running = state.busy || !!p?.busy;
  if (!running) {
    return (
      <div class="clipprog idle">
        <div class="stage">
          <span class="i dim" props={{ innerHTML: state.error ? "&#xf071;" : "&#xf00c;" }} /> {state.status}
        </div>
      </div>
    );
  }
  const phase = p?.busy ? p.phase ?? "prompt" : "";
  const at = p?.at ?? 0;
  const of = Math.max(p?.of ?? 1, 1);
  const frac = Math.min(at / of, 1);
  const pct = Math.round(100 * frac);
  const secs = p?.busy ? p.seconds ?? 0 : null;
  let stage = p?.waiting ? `waiting: ${p.waiting} ahead` : "starting";
  let extra = "";
  if (phase === "prompt") stage = "reading the prompt";
  else if (phase === "tokens") {
    const fps = state.info?.speech ? 12.5 : 25;
    stage = `${state.info?.speech ? "speaking" : "composing"}: ${mss(at / fps)} of up to ${mss(of / fps)}`;
    extra = "the language model, a frame at a time";
  } else if (phase === "flow") {
    stage = `rendering the sound: step ${at}/${of}`;
  } else if (phase === "decode") stage = `decoding window ${at}/${of}`;
  const phases = state.info?.speech ? PHASES.filter((x) => x !== "flow") : PHASES;
  const idx = phases.indexOf(phase);
  return (
    <div class="clipprog">
      <div class="bar"><div class="fill" style={{ width: `${Math.max(pct, 2)}%` }} /></div>
      <div class="stage">
        <span class="i" props={{ innerHTML: "&#xf110;" }} /> {stage}{" "}
        <span class="hint">{`${pct}%${secs !== null ? ` · ${fmtT(secs)} elapsed` : ""}${extra ? ` · ${extra}` : ""}`}</span>
      </div>
      <div class="phase">
        {phases.map((x, i) => <span class={{ on: i === idx, done: idx > i }}>{x}</span>)}
      </div>
    </div>
  );
}
