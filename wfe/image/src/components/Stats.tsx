/** The metric strip: the model's state, its defaults, the last request, the session. */
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
  const d = info?.defaults;
  const h = state.history;
  const last = h[0];
  const loaded = (info?.loras ?? []).filter((l) => l.loaded !== null);
  const engine = info?.loading
    ? ["Loading", "the model is loading"]
    : p?.busy
      ? [p.loading ? "Reloading" : "Running", p.loading ? "merging LoRAs" : `step ${p.at ?? 0}/${p.of ?? "?"}${p.waiting ? ` · ${p.waiting} waiting` : ""}`]
      : ["Idle", p?.waiting ? `${p.waiting} waiting` : "ready"];
  const wh = h.reduce((a, x) => a + (x.wh ?? 0), 0);
  return (
    <div class="stats">
      {card("&#xf04b;", "engine", engine[0] ?? "", engine[1] ?? "")}
      {card("&#xf013;", "defaults", d ? `${d.steps} steps` : "—", d ? `${d.width}×${d.height} · ${d.sampler} · cfg ${d.cfg}` : "")}
      {card("&#xf0c5;", "LoRAs merged", loaded.length ? String(loaded.length) : "none",
            loaded.map((l) => `${l.id} ×${l.loaded}`).join(", ") || `${(info?.loras ?? []).length} available`)}
      {card("&#xf017;", "last picture", last ? fmtT(last.seconds) : "—",
            last ? `${last.width}×${last.height} · ${last.steps} steps${last.wh != null ? ` · ${last.wh.toFixed(2)} Wh` : ""}` : "nothing yet")}
      {card("&#xf0e7;", "pictures", `${h.length}`, wh ? `${wh.toFixed(1)} Wh in all` : "")}
    </div>
  );
}
