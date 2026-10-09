/** What the GPU is doing on the request running now: stage, bar, seconds (H3's ClipProgress). */
import { jsx } from "../jsx.js";
import { state } from "../state.js";
import { fmtT } from "../util.js";

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
  const of = p?.of ?? 0;
  const pics = p?.pictures ?? 1;
  const frac = of ? (((p?.picture ?? 1) - 1) * of + (p?.at ?? 0)) / (pics * of) : 0;
  const pct = p?.loading ? 3 : Math.round(3 + 97 * frac);
  const stage = p?.loading
    ? "loading the model with those LoRAs"
    : p?.busy && p.at
      ? `picture ${p.picture}/${pics} · step ${p.at}/${of}`
      : p?.busy ? "encoding the prompt" : p?.waiting ? `waiting: ${p.waiting} ahead` : "starting";
  const secs = p?.busy ? p.seconds ?? 0 : null;
  const perStep = p?.at && secs ? secs / ((((p.picture ?? 1) - 1) * of) + p.at) : null;
  const left = perStep && of ? perStep * (pics * of - ((((p?.picture ?? 1) - 1) * of) + (p?.at ?? 0))) : null;
  return (
    <div class="clipprog">
      <div class="bar"><div class="fill" style={{ width: `${pct}%` }} /></div>
      <div class="stage">
        <span class="i" props={{ innerHTML: "&#xf110;" }} /> {stage}{" "}
        <span class="hint">
          {`${pct}%${secs !== null ? ` · ${fmtT(secs)} elapsed` : ""}${perStep ? ` · ${perStep.toFixed(2)} s/step` : ""}${left ? ` · ${fmtT(left)} left` : ""}`}
        </span>
      </div>
    </div>
  );
}
