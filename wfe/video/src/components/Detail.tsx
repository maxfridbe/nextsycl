/** One clip: its settings, timings and prompt. Same panel whether it has rendered or not.
 *  Rendered directly under the strip, so the thing you clicked and what it says are together. */
import { jsx } from "../jsx.js";
import { render, state } from "../state.js";
import { navigate } from "../router.js";
import type { Job } from "../types.js";
import { fmtT } from "../util.js";
import { rpc } from "../api.js";

/** A made clip again at 2560x1440 (Open-H3-Regenerate-2K-IC on the Regenerate-2K IC model): queued as its own clip,
 *  the draft's prompt as the caption, the same label with " 2K" */
async function regen2k(name: string, j: Job): Promise<void> {
  regenSent = "sending…";
  render();
  try {
    const r = await rpc("generate", {
      prompt: j.prompt ?? "", regen2k: `${name}.mp4`, seed: j.seed ?? 0, queue: true,
      label: j.label ? `${j.label} 2K`.slice(0, 80) : undefined, title: `${j.title || j.label || name} (2K)`,
      project: j.project, batch: j.batch,
    });
    regenSent = r.queued ? `2K queued at position ${r.position ?? "?"}` : "2K started";
    state.error = null;
  } catch (e) {
    regenSent = "";
    state.error = `2K regeneration: ${String(e)}`;
  }
  render();
}

/** What the last Regenerate 2K did, shown beside the button */
let regenSent = "";

const row = (icon: string, k: string, v: unknown) => (
  <tr>
    <td class="dk"><span class="i dim" props={{ innerHTML: icon }} /> {k}</td>
    <td>{v === null || v === undefined || v === "" ? "—" : String(v)}</td>
  </tr>
);

const settings = (j: Job) => [
  row("&#xf02b;", "title", j.title),
  row("&#xf02b;", "label", j.label),
  row("&#xf017;", "seconds", j.seconds),
  row("&#xf1de;", "steps", j.steps),
  row("&#xf0e4;", "pace", j.speed === "pika-flex" || (j.knots && j.sampler === "lcm") ? "Pika Flex" : "standard"),
  row("&#xf074;", "seed", j.seed),
  row("&#xf19d;", "encoder", j.te),
  row("&#xf085;", "engine", j.engine ?? "Q4_K_M"),
  row("&#xf0b2;", "canvas", `${j.width ?? 640}×${j.height ?? 480}`),
  row("&#xf0c1;", "first frame", j.first_frame ?? "none"),
  row("&#xf03a;", "camera", j.camera),
  row("&#xf07b;", "project", j.project ? `${j.project}${j.batch ? ` · ${j.batch}` : ""}` : null),
  row("&#xf00e;", "upscale", j.upscale ? `${j.upscale}×` : null),
  row("&#xf065;", "regenerated", j.regen2k ? `${j.regen2k} at 2K (${j.regen_sampler ?? "up3"})` : null),
  row("&#xf0d0;", "loras", (j.loras ?? []).map((x) => x.split("/").pop()).join(", ")),
];

export function Detail() {
  const d = state.detail;
  if (!d) return <div />;
  const j: Job = d.kind === "job" ? (d.data.job ?? ({} as Job)) : d.data;
  const times = d.kind === "job" ? (d.data.times ?? {}) : {};
  const title = d.kind === "job" ? d.name : (j.title || j.label || "queued clip");
  return (
    <div class="detail">
      <div class="row dhead">
        <b>{title}</b>
        {d.kind === "clip" ? <span class="hint">{` · ${j.state ?? "queued"}`}</span> : null}
        {d.kind === "job"
          ? <button
              type="button"
              class="tbtn"
              on={{ click: () => {
                state.video = { file: `${d.name}.mp4`, poster: `thumb/${d.name}.webp`, playing: true };
                render();
              } }}
            >Play</button>
          : null}
        {d.kind === "job" && !j.regen2k
          ? <button type="button" class="tbtn" attrs={{ title: "made again at 2560x1440 by Open-H3-Regenerate-2K-IC: faces, lip sync and lettering kept, its sound kept (about half an hour)" }}
              on={{ click: () => void regen2k(d.name, j) }}>Regenerate 2K</button>
          : null}
        {regenSent ? <span class="hint">{regenSent}</span> : null}
        <button type="button" class="tbtn" on={{ click: () => navigate({ clip: null }) }}>Close</button>
      </div>
      <div class="dtables">
        <table><tbody>{settings(j)}</tbody></table>
        <table>
          <tbody>
            {Object.entries(times).map(([k, v]) => (
              <tr>
                <td class="dk">{k}</td>
                <td class="num">{fmtT(v)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {j.dialogue ? <div class="hint dlg">{`“${j.dialogue}”`}</div> : null}
      <pre>{j.prompt ?? ""}</pre>
    </div>
  );
}

/** Sits above the timeline. Shows the clip's own thumbnail with a play button over it and
 *  only creates the <video> element once that button is pressed - so nothing ever autoplays
 *  and a click costs no bandwidth. */
export function Player() {
  const v = state.video;
  if (!v) return <div />;
  const close = () => { state.video = null; render(); };
  return (
    <div class="player">
      <div class="row dhead">
        <b>{v.file}</b>
        <a class="tbtn" attrs={{ href: `out/${encodeURIComponent(v.file)}`, download: true }}>download</a>
        <button type="button" class="tbtn" on={{ click: close }}>close</button>
      </div>
      {v.playing
        ? <video attrs={{ controls: true, autoplay: true, src: `out/${encodeURIComponent(v.file)}` }} />
        : (
          <div class="poster" on={{ click: () => { state.video = { ...v, playing: true }; render(); } }}>
            {v.poster
              ? <img attrs={{ src: v.poster, alt: `play ${v.file}` }} />
              : <div class="ph" />}
            <button type="button" class="playbtn" attrs={{ title: "play" }}>&#9654;</button>
          </div>
        )}
    </div>
  );
}
