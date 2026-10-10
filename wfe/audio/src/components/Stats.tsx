/** The metric strip: the engine's state, its defaults, the last song or speech, those made here. */
import { jsx } from "../jsx.js";
import { state } from "../state.js";
import { fmtT, mss } from "../util.js";

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
  const d = info?.defaults;
  const speech = !!info?.speech;
  const h = state.history.filter((s) => (s.cloned !== undefined) === speech);
  const last = h[0];
  const what = speech ? "speech" : "song";
  const engine = p?.busy
    ? ["Running", `${p.phase ?? ""} ${p.at ?? 0}/${p.of ?? "?"}${p.waiting ? ` · ${p.waiting} waiting` : ""}`]
    : ["Idle", p?.waiting ? `${p.waiting} waiting` : info?.report?.[0] ?? "ready"];
  const wh = h.reduce((a, x) => a + (x.wh ?? 0), 0);
  const sound = h.reduce((a, x) => a + x.seconds, 0);
  return (
    <div class="stats">
      {card("&#xf04b;", "engine", engine[0] ?? "", engine[1] ?? "")}
      {card("&#xf013;", "defaults", d ? `${mss(d.seconds)}` : "—", d ? `${speech ? "" : `${d.steps} steps · cfg ${d.cfg} · `}up to ${mss(d.max_seconds)} · ${d.rate / 1000} kHz` : "")}
      {card("&#xf017;", `last ${what}`, last ? mss(last.seconds) : "—",
            last ? `made in ${fmtT(last.took ?? 0)}${last.took ? ` (${(last.seconds / last.took).toFixed(2)}×)` : ""}${last.wh != null ? ` · ${last.wh.toFixed(2)} Wh` : ""}` : "nothing yet")}
      {card(speech ? "&#xf130;" : "&#xf001;", speech ? "made" : "songs", `${h.length}`, `${mss(sound)} in all${wh ? ` · ${wh.toFixed(1)} Wh` : ""}`)}
    </div>
  );
}
