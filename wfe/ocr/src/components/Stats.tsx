/** The metric strip: the model's state, the reading running or the last one, the session. */
import { jsx } from "../jsx.js";
import { state } from "../state.js";
import { fmtT } from "../util.js";

const card = (icon: string, k: string, v: string, sub: string) => (
  <div class="stat">
    <div class="k"><span class="i dim" props={{ innerHTML: icon }} /> {k}</div>
    <div class="v">{v}</div>
    <div class="s">{sub}</div>
  </div>
);

export function Stats() {
  const info = state.info;
  const p = state.progress;
  const h = state.history;
  const last = h[0];
  const engine = info?.loading
    ? ["Loading", "the model is loading"]
    : p?.busy
      ? ["Reading", `${p.pages ?? 1} page${(p.pages ?? 1) === 1 ? "" : "s"} · ${p.tokens ?? 0} tokens${p.waiting ? ` · ${p.waiting} waiting` : ""}`]
      : ["Idle", p?.waiting ? `${p.waiting} waiting` : "ready"];
  const speed = p?.busy && p.seconds ? `${((p.tokens ?? 0) / p.seconds).toFixed(0)} tok/s` : last?.tok_s ? `${last.tok_s.toFixed(0)} tok/s` : "—";
  const wh = h.reduce((a, x) => a + (x.wh ?? 0), 0);
  const tokens = h.reduce((a, x) => a + x.tokens, 0);
  return (
    <div class="stats">
      {card("&#xf04b;", "engine", engine[0] ?? "", engine[1] ?? "")}
      {card("&#xf0e4;", "writing", speed, p?.busy ? `${fmtT(p.seconds)} so far` : last ? `last: ${fmtT(last.seconds)}` : "nothing yet")}
      {card("&#xf017;", "last reading", last ? fmtT(last.seconds) : "—",
            last ? `${last.pages} page${last.pages === 1 ? "" : "s"} · ${last.tokens} tokens${last.wh != null ? ` · ${last.wh.toFixed(2)} Wh` : ""}` : "")}
      {card("&#xf02d;", "readings", `${h.length}`, `${tokens} tokens${wh ? ` · ${wh.toFixed(1)} Wh` : ""}`)}
    </div>
  );
}
