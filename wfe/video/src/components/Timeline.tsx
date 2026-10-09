/** The edit, as a strip: one track per camera, clip width proportional to duration, blanks
 *  where a camera is not shooting.
 *
 *  Reading order down the panel is deliberate and matches how you watch a render: what the
 *  GPU is doing on this one clip, then how far the whole project has come, then the strip,
 *  then the clip you picked. */
import { jsx } from "../jsx.js";
import { api } from "../api.js";
import { cameraName, setCameraName, state } from "../state.js";
import { navigate } from "../router.js";
import type { Scene, TimelineClip } from "../types.js";
import { fmtMS, fmtT } from "../util.js";
import { ClipProgress } from "./Progress.js";
import { Detail } from "./Detail.js";
import { Panel } from "./Panel.js";

function exportScene(): void {
  if (state.timelineProject) {
    location.href = `/api/scene?project=${encodeURIComponent(state.timelineProject)}`;
  }
}

async function importScene(file: File): Promise<void> {
  try {
    const scene = JSON.parse(await file.text()) as Scene;
    const suggested = scene.project || file.name.replace(/\.scene\.json$/, "");
    const name = window.prompt("Queue this scene as project:", suggested);
    if (name === null) return;
    const r = await api.importScene(scene, name, true);
    if (r.error) {
      window.alert(`import failed: ${r.error}`);
      return;
    }
    window.alert(`imported ${r.imported} clips as "${r.project}" (held — resume when ready)`);
    navigate({ tab: "jobs", project: r.project ?? name, clip: null });
  } catch (e) {
    window.alert(`import failed: ${String(e)}`);
  }
}

const tool = (label: string, onclick: () => void) => (
  <button type="button" class="tbtn" on={{ click: onclick }}>{label}</button>
);

/** Clips laid end to end on one shared time axis, so two cameras line up. */
interface Placed { clip: TimelineClip; at: number }

function layout(clips: TimelineClip[]): { placed: Placed[]; total: number } {
  let at = 0;
  const placed = clips.map((clip) => {
    const put = { clip, at };
    at += clip.seconds || 0;
    return put;
  });
  return { placed, total: at };
}

function ClipCard(props: { placed: Placed; px: number }) {
  const { clip, at } = props.placed;
  const alt = clip.dialogue || "(no dialogue)";
  return (
    <div
      class={{ tlc: true, [clip.state]: true, sel: state.selected === clip.n }}
      dataset={{ n: String(clip.n) }}
      style={{ left: `${Math.round(at * props.px)}px`, width: `${Math.max(26, Math.round((clip.seconds || 0) * props.px))}px` }}
      attrs={{ title: `${clip.label}\n${alt}` }}
      on={{ click: () => navigate({ clip: clip.n }) }}
    >
      {clip.thumb
        ? <img attrs={{ src: clip.thumb, alt, loading: "lazy" }} />
        : <div class="ph">{clip.state === "running" ? "●" : "…"}</div>}
      <span class="n">{clip.n}</span>
      <span class="s">{clip.seconds ? `${clip.seconds.toFixed(1)}s` : ""}</span>
    </div>
  );
}

/** One track. The name is editable because "A" and "B" stop meaning anything the moment
 *  a project has more than one setup. */
function CameraTrack(props: { camera: string; placed: Placed[]; px: number; project: string | null }) {
  const mine = props.placed.filter(({ clip }) => (clip.camera || "—") === props.camera);
  const seconds = mine.reduce((a, { clip }) => a + (clip.seconds || 0), 0);
  const name = cameraName(props.project, props.camera);
  const rename = () => {
    const next = window.prompt(`Name for ${props.camera === "—" ? "this camera" : `camera ${props.camera}`}:`, name);
    if (next !== null) setCameraName(props.project, props.camera, next);
  };
  return (
    <div class="tlrow">
      <span class="tllab" attrs={{ title: "click to rename this camera" }} on={{ click: rename }}>
        {name}
        <small>{` ${mine.length} · ${fmtMS(seconds)}`}</small>
      </span>
      {mine.map((p) => <ClipCard placed={p} px={props.px} />)}
    </div>
  );
}

/** How far the whole project has come, measured in finished seconds of film. */
function ProjectProgress(props: { clips: TimelineClip[]; total: number }) {
  const { clips, total } = props;
  const done = clips.filter((c) => c.state === "done");
  const doneSeconds = done.reduce((a, c) => a + (c.seconds || 0), 0);
  const heldCount = clips.filter((c) => c.state === "held").length;
  const running = clips.find((c) => c.state === "running");
  const pct = total > 0 ? Math.round((doneSeconds / total) * 100) : 0;
  const ratio = state.summary?.project_stats?.ratio ?? null;
  const etaSeconds = ratio ? (total - doneSeconds) * ratio : null;
  return (
    <div class="tlprog">
      <div class="bar">
        <div class="fill" style={{ width: `${pct}%` }} />
        {running
          ? <div
              class="now"
              style={{ left: `${Math.round(((doneSeconds + (running.seconds || 0) / 2) / Math.max(total, 1)) * 100)}%` }}
              attrs={{ title: running.label }}
            />
          : null}
      </div>
      <div class="hint">
        {`whole project: ${done.length}/${clips.length} clips · ${fmtMS(doneSeconds)} of ${fmtMS(total)} · ${pct}%`}
        {heldCount ? ` · ${heldCount} held` : ""}
        {etaSeconds ? ` · ~${fmtT(etaSeconds)} of GPU left` : ""}
      </div>
    </div>
  );
}

export function Timeline() {
  const clips = state.timelineClips;
  const project = state.timelineProject;
  if (!project) return <div />;

  const px = state.pxPerSecond;
  const { placed, total } = layout(clips);
  const cameras: string[] = [];
  for (const { clip } of placed) {
    const k = clip.camera || "—";
    if (!cameras.includes(k)) cameras.push(k);
  }
  const done = clips.filter((c) => c.state === "done").length;

  const picker = (
    <select
      class="sel"
      on={{ change: (e: Event) => navigate({ project: (e.target as HTMLSelectElement).value, clip: null }) }}
    >
      {(state.status?.all_projects ?? [project]).map((n) => (
        <option attrs={{ value: n, selected: n === project }}>{n}</option>
      ))}
    </select>
  );

  const hint = `${done} of ${clips.length} rendered · ${fmtMS(total)} of film · `
    + `${cameras.length} camera${cameras.length === 1 ? "" : "s"}`;

  return (
    <Panel
      id="timeline"
      icon="&#xf1de;"
      title="Timeline"
      hint={hint}
      tools={[
        picker,
        tool("export JSON", exportScene),
        tool("import JSON", () => document.getElementById("sceneFile")?.click()),
        tool("types.ts", () => { location.href = "/api/types.ts"; }),
      ]}
    >
      <ClipProgress />
      <ProjectProgress clips={clips} total={total} />
      <div class="tl">
        <div class="tlinner" style={{ width: `${Math.round(total * px) + 40}px` }}>
          {cameras.map((k) => (
            <CameraTrack camera={k} placed={placed} px={px} project={project} />
          ))}
        </div>
      </div>
      <Detail />
      <input
        attrs={{ type: "file", id: "sceneFile", accept: "application/json,.json" }}
        style={{ display: "none" }}
        on={{
          change: (e: Event) => {
            const f = (e.target as HTMLInputElement).files?.[0];
            if (f) void importScene(f);
            (e.target as HTMLInputElement).value = "";
          },
        }}
      />
    </Panel>
  );
}
