/** The page: the host strip and the tabs - Status (the GPUs with what holds them and what is vouched for them, the
 *  models by kind with enable / pin, the front ends, the actions' output) and API. */
import { jsx } from "../jsx.js";
import { act, autoPull, copy, maintenance, render, setTab, ui } from "../main.js";
import type { Available, Card, Model, Proc, Service } from "../types.js";
import { Api } from "./Api.js";
import { ConfigView } from "./Config.js";
import { DrivesPanel, ModelDisk } from "./Drives.js";
import { Spinner } from "./Spinner.js";

const gib = (mb: number) => (mb / 1024).toFixed(1);
const short = (n: string) => n.replace("Intel(R) Arc(TM) ", "").replace(" Graphics", "");
/** a link to a service on this box, as the browser reaches it */
const at = (port: number, path = "/") => `${location.protocol}//${location.hostname}:${port}${path}`;

const COLORS: Record<string, string> = { llm: "#06b6d4", image: "#f59e0b", video: "#ec4899", audio: "#8b5cf6", model3d: "#22c55e" };
const color = (k?: string) => COLORS[k ?? ""] ?? "#64748b";
const KINDS: { kind: Model["kind"]; title: string; icon: string }[] = [
  { kind: "llm", title: "Chat", icon: "&#xf086;" },
  { kind: "image", title: "Images", icon: "&#xf03e;" },
  { kind: "audio", title: "Audio", icon: "&#xf001;" },
  { kind: "video", title: "Video", icon: "&#xf03d;" },
  { kind: "model3d", title: "3D", icon: "&#xf1b2;" },
];

/** A vouched figure: the weights, then the headroom - the working memory it needs to run without erroring (KV for its
 *  context, arenas, a song's or an image's activations) - in parentheses with its swatch: the model's color, its
 *  stripes the other way (as in the card's vouch bar) */
function Alloc(mb: number, headroom: number, kind: string, given: boolean, unit = " GiB") {
  const h = Math.min(headroom, mb);
  const title = h > 0
    ? `${gib(mb)} GiB vouched: ${gib(mb - h)} for its weights + ${gib(h)} headroom (${given ? "from its model card" : "estimate: vouched less its files"})`
    : `${gib(mb)} GiB vouched (no headroom known apart: its files do not all sit on the card)`;
  return (
    <span attrs={{ title }}>{gib(mb - h)}{unit}{h > 0
      ? <span class="hr"> (+<span class="sw hatch rev" style={{ "--c": color(kind) } as Record<string, string>} />{gib(h)}{unit} headroom)</span>
      : null}</span>
  );
}

function procName(p: Proc): string {
  if (p.program !== "nextsycl") return p.program;
  return [p.kind, p.command, p.model].filter((x) => x).join(" ");
}

function CardView(c: Card) {
  const total = Math.max(c.vram_total_mb, 1);
  const free = c.vram_total_mb - c.vram_used_mb;
  return (
    <div class={{ card: true, over: c.overbooked }}>
      <h3>
        <span class={{ dot: true, busy: c.busy_pct > 5, off: c.vram_used_mb < 512 }} />
        {short(c.name)} <span class="idx">GPU {c.index} · {c.pci}</span>
      </h3>
      <div class="meter" attrs={{ title: `${gib(c.vram_used_mb)} of ${gib(c.vram_total_mb)} GiB in use` }}>
        {c.procs.map((p) => <i style={{ width: `${(p.vram_mb / total) * 100}%`, background: color(p.kind) }} />)}
      </div>
      <div class="meter vouch" attrs={{ title: `${gib(c.vouched_mb)} GiB vouched for by the enabled models` }}>
        {/* each model's weights, then its headroom: the same color, the stripes the other way */}
        {c.vouched.map((v) => {
          const h = Math.min(v.headroom_mb ?? 0, v.mb);
          const live = ["loaded", "busy"].includes(v.state);
          const seg = (mb: number, rev: boolean) => mb > 0
            ? <i class={{ hatch: true, rev, live }} attrs={{ title: `${v.model}: ${gib(mb)} GiB ${rev ? "headroom" : "weights"}` }}
                style={{ width: `${Math.min(100, (mb / total) * 100)}%`, "--c": color(v.kind) } as Record<string, string>} />
            : null;
          return [seg(v.mb - h, false), seg(h, true)];
        })}
      </div>
      <div class="kv">
        <span>in use <b>{gib(c.vram_used_mb)}</b> / {gib(c.vram_total_mb)} GiB · <span class={free < 1536 ? "bad" : ""}>{gib(free)} free</span></span>
        <span class={c.overbooked ? "warn" : ""}>vouched <b>{gib(c.vouched_mb)}</b> GiB{c.vouched_headroom_mb > 0
          ? <span class="hint"> ({gib(c.vouched_headroom_mb)} of it headroom)</span> : null}{c.overbooked ? " - overbooked: the first to load holds the card" : ""}</span>
        <span>busy <b>{Math.round(c.busy_pct)}%</b></span>
        {c.power_w != null ? <span>power <b>{Math.round(c.power_w)} W</b>{c.power_cap_w ? ` / ${c.power_cap_w}` : ""}</span> : null}
        {c.temp_pkg != null ? <span>temp <b>{Math.round(c.temp_pkg)}°C</b>{c.temp_vram != null ? ` · mem ${Math.round(c.temp_vram)}°C` : ""}</span> : null}
        {c.pcie?.cur ? <span>PCIe <b>{c.pcie.cur.gen ? `${c.pcie.cur.gen}.0` : "?"} x{c.pcie.cur.width}</b></span> : null}
      </div>
      <ul class="procs">
        {c.procs.map((p) => {
          // a loaded model's process: what it is vouched for beside what it holds now
          const v = c.vouched.find((x) => x.model === p.model_id && ["loaded", "busy"].includes(x.state));
          return (
          <li attrs={{ title: p.args ?? `pid ${p.pid}` }}>
            <span class="sw" style={{ background: color(p.kind) }} />
            <span class="what">{procName(p)}{p.port ? <span class="hint"> :{p.port}</span> : null}
              {v && v.headroom_mb > 0 ? <span class="hint"> · vouched {Alloc(v.mb, v.headroom_mb, v.kind, v.headroom_given)}</span> : null}</span>
            <span class="mb">{gib(p.vram_mb)} GiB</span>
          </li>
          );
        })}
        {c.vouched.filter((v) => !["loaded", "busy"].includes(v.state)).map((v) => (
          <li class="dim">
            <span class="sw hatch" style={{ "--c": color(v.kind) } as Record<string, string>} />
            <span class="what">{v.model}{v.of && v.of > 1 ? <span class="hint"> (one of {v.of} {v.kind === "llm" ? "chat models" : "video engines"}: one runs at a time)</span> : null}
              <span class="hint"> {v.state}{v.measured ? "" : " · estimate"}</span></span>
            <span class="mb">{Alloc(v.mb, v.headroom_mb ?? 0, v.kind, v.headroom_given)}</span>
          </li>
        ))}
        {!c.procs.length && !c.vouched.length ? <li class="dim"><span class="what hint">free, nothing enabled here</span></li> : null}
      </ul>
    </div>
  );
}

const STATE_TEXT: Record<string, string> = {
  disabled: "disabled", unloaded: "enabled · unloaded", loading: "loading…", loaded: "loaded", busy: "working",
  // a video model the daemon has, its weights not on a card (they load with the first clip)
  ready: "enabled · unloaded (daemon up)",
};

function GpuPicker(m: Model) {
  const cards = ui.state?.cards ?? [];
  // an enabled model's cards are its own (changed at once); a disabled one's are what enable will offer
  const pick = m.enabled ? m.gpus : (ui.pick[m.id] ??= [...m.gpus]);
  const loaded = ["loaded", "busy", "loading"].includes(m.state);
  // one copy spread over every card: there is nothing to choose
  const fixed = !!m.spans;
  return (
    <span class="gpus" attrs={{ title: fixed ? "spread over every card: it does not run on fewer"
      : (m.kind === "image" || m.kind === "audio" ? "the cards it may load on (the first with room)" : "the cards it may use")
      + (m.enabled ? " - click to change (more than fit is allowed; a loaded model moved off its card unloads)" : "") }}>
      {cards.map((c) => (
        <label class={{ gpu: true, on: fixed || pick.includes(c.index), fixed }}>
          <input attrs={{ type: "checkbox", name: `g-${m.id}`, disabled: fixed || !!ui.pending[m.id] }} props={{ checked: fixed || pick.includes(c.index) }}
            on={{ change: (e: Event) => {
              const on = (e.target as HTMLInputElement).checked;
              const next = on ? [...new Set([...pick, c.index])].sort() : pick.filter((g) => g !== c.index);
              if (!next.length) {
                (e.target as HTMLInputElement).checked = true;
                ui.error = `${m.id}: a model needs at least one card (disable it instead)`;
                render();
                return;
              }
              if (!m.enabled) {
                ui.pick[m.id] = next;
                render();
                return;
              }
              if (loaded && pick.some((g) => !next.includes(g)) && !window.confirm(`${m.id} is loaded on GPU ${pick.join(", ")}: it unloads now and loads on GPU ${next.join(", ")} at its next request.`)) {
                render();
                return;
              }
              void act("gpus", m.id, { gpus: next });
            } }} />
          {short(c.name).replace("Pro ", "")}
        </label>
      ))}
    </span>
  );
}

function ModelRow(m: Model) {
  const p = ui.pending[m.id];
  const spin = (what: string) => p?.what === what;
  // weights on a card (a video model "ready" is not: the daemon is up, its weights load with the first clip)
  const loaded = ["loaded", "busy"].includes(m.state);
  const btn = (what: string, label: string, cls: string, title: string, extra: Record<string, unknown> = {}) => (
    <button class={cls} attrs={{ disabled: !!p || m.state === "loading", title }}
      on={{ click: () => void act(what, m.id, extra) }}>
      {spin(what) ? <span class="spin" /> : null}{label}
    </button>
  );
  const vouch = Object.entries(m.vouch_mb).map(([g, mb], i) =>
    [i ? ", " : "", `GPU ${g}: `, Alloc(mb, m.headroom_mb?.[g] ?? 0, m.kind, m.headroom_given)]);
  const page = m.kind === "llm" ? at(m.page) : at(m.page);
  return (
    <div class={{ mrow: true, off: !m.enabled }}>
      <span class={{ dot: true, off: !loaded, busy: m.state === "busy" || m.state === "loading" }} />
      <div class="mname">
        <div class="name">{m.title ?? m.id}</div>
        <div class="sub">{m.id} · <span attrs={{ title: m.measured ? "as seen while it ran" : "an estimate from its files, until it has run" }}>{vouch}{m.measured ? "" : " (est.)"}</span></div>
      </div>
      <div class="mstate"><span class={{ tag: true, [m.state]: true }}>{STATE_TEXT[m.state] ?? m.state}</span>
        {m.pinned ? <span class="hint" attrs={{ title: "never stopped for room" }}> · pinned</span> : null}
        {m.held_s ? <span class="hint" attrs={{ title: "held by rsvp_s or X-Keep-Up-S: not stopped for room" }}> · held {Math.ceil(m.held_s / 60)} min</span> : null}</div>
      <div class="acts">
        <span class="onoff" attrs={{ title: m.enabled ? "enabled: it loads on its first request and unloads after the idle time" : "disabled: off the API (and Open WebUI's list)" }}>
          <label class={{ on: !m.enabled }}>
            <input attrs={{ type: "radio", name: `en-${m.id}`, disabled: !!p || m.state === "loading" }} props={{ checked: !m.enabled }}
              on={{ change: () => { if (m.enabled) void act("disable", m.id, m.kind === "video" ? { force: false } : {}); } }} /> disabled
          </label>
          <label class={{ on: m.enabled }}>
            <input attrs={{ type: "radio", name: `en-${m.id}`, disabled: !!p || m.state === "loading" }} props={{ checked: m.enabled }}
              on={{ change: () => { if (!m.enabled) void act("enable", m.id, { gpus: ui.pick[m.id] ?? m.gpus }); } }} /> enabled
          </label>
          {p && (p.what === "enable" || p.what === "disable") ? <span class="spin" /> : null}
        </span>
        {GpuPicker(m)}
        {m.enabled
          ? [
              m.pinned
                ? btn("unpin", "unpin", "ghost", "pinned: loaded and never stopped for room, also after a restart. Unpin: it may be stopped for room and unloads when idle")
                : btn("pin", "pin", "ghost", "load it now and keep it: never stopped for room, loaded again after a restart. Otherwise requests load it and idle or a need for room unloads it"),
              // its own page, loaded or not, through the front door (/open/ID): an image, audio or 3D model's is its server's,
              // so the model starts first; a video model's is the studio's
              <a class="btn" attrs={{ href: m.kind !== "llm" ? `/open/${encodeURIComponent(m.id)}` : page, target: "_blank",
                title: m.kind === "llm" ? "Open WebUI" : m.kind === "video" ? "the video studio"
                  : loaded ? "its page" : "its page: the model starts first (a minute), then the page opens" }}>
                <span class="i" props={{ innerHTML: "&#xf08e;" }} /> page
              </a>,
            ]
          : null}
      </div>
      {ModelDisk(m)}
    </div>
  );
}

function Models() {
  const st = ui.state;
  if (!st) return <section class="panel"><div class="pbody"><Spinner label="Loading the models…" /></div></section>;
  return (
    <section class="panel">
      <div class="phead"><span class="i" props={{ innerHTML: "&#xf1b3;" }} /><span class="ptitle">Models</span>
        <span class="hint">enabled models load on their first request and unload after {st.idle_minutes} min idle · disabled ones are off Open WebUI and the API</span></div>
      <div class="pbody">
        {KINDS.map((k) => {
          const list = st.models.filter((m) => m.kind === k.kind);
          if (!list.length) return null;
          const on = list.filter((m) => m.enabled).length;
          return (
            <div class="kind">
              <h4 style={{ color: color(k.kind) }}><span class="i" props={{ innerHTML: k.icon }} /> {k.title}
                <span class="hint"> {on} of {list.length} enabled</span></h4>
              {list.map(ModelRow)}
            </div>
          );
        })}
      </div>
    </section>
  );
}

const GIB = 1024 ** 3;
const size = (b: number) => `${(b / GIB).toFixed(1)} GiB`;
const KIND_TITLE: Record<string, string> = { llm: "Chat", image: "Images", audio: "Audio", video: "Video", model3d: "3D" };
const eta = (s: number | null) => s == null ? "" : s >= 3600 ? `${(s / 3600).toFixed(1)} h` : s >= 60 ? `${Math.round(s / 60)} min` : `${Math.round(s)} s`;

/** A model a card offers that is not here: its size and where it would go, a Download button - or its download's
 *  progress with a cancel, or the last one's error */
function AvailableRow(a: Available) {
  const p = ui.pending[a.id];
  const d = a.pull;
  const last = (ui.state?.pulls ?? []).find((x) => x.model === a.id);
  const failed = !d && last && last.state === "failed" ? last : null;
  const going = !!d && !["done", "failed", "cancelled"].includes(d.state);
  return (
    <div class="mrow off">
      <span class={{ dot: true, off: !going, busy: going }} />
      <div class="mname">
        <div class="name">{a.title ?? a.id}</div>
        <div class="sub">{a.id} · {size(a.bytes)} · to {a.drive}{a.fits ? "" : <span class="bad"> - not enough room ({size(a.free)} free)</span>}</div>
        {a.runs_here === false ? <div class="hint bad">won't run here: {a.why_not}</div> : null}
        {going && d
          ? <div>
              <div class="meter" attrs={{ title: `${size(d.bytes_done)} of ${size(d.bytes_total)}` }}>
                <i style={{ width: `${d.pct}%`, background: color(a.kind) }} />
              </div>
              <div class="hint">{d.state === "queued" ? "queued behind another download to the same folder"
                : d.state === "verifying" ? "checking its SHA-256…"
                : `${d.pct}% · ${size(d.bytes_done)} of ${size(d.bytes_total)}${d.speed > 0 ? ` · ${(d.speed / 1e6).toFixed(0)} MB/s` : ""}${d.eta_s ? ` · ${eta(d.eta_s)} left` : ""}`}</div>
            </div>
          : null}
        {failed ? <div class="hint bad" attrs={{ title: failed.error ?? "" }}>the last download failed: {failed.error}</div> : null}
      </div>
      <div class="mstate"><span class="tag">{going ? (d?.state ?? "downloading") : a.runs_here === false ? "won't run here" : "not downloaded"}</span></div>
      <div class="acts">
        {going
          ? <button class="ghost" attrs={{ disabled: !!p, title: "stop it: what was fetched is kept, a download again resumes it" }}
              on={{ click: () => void act("pull_cancel", a.id) }}>{p ? <span class="spin" /> : null}cancel</button>
          : <button attrs={{ disabled: !!p || !a.fits || a.runs_here === false,
                             title: a.runs_here === false ? `these cards cannot run it (${a.why_not}); system.models.pull still fetches it`
                               : a.fits ? `download ${size(a.bytes)} into ${a.folder}, check it, and register it (enabled)` : "no models folder has the room (NS_MODEL_DIRS adds folders)" }}
              on={{ click: () => void act("pull", a.id) }}>{p ? <span class="spin" /> : <span class="i" props={{ innerHTML: "&#xf019;" }} />} {failed ? "retry" : "download"}</button>}
      </div>
    </div>
  );
}

function NotDownloaded() {
  const st = ui.state;
  const list = st?.available ?? [];
  if (!st || !list.length) return null;
  const kinds = [...new Set(list.map((a) => a.kind))];
  return (
    <section class="panel">
      <div class="phead"><span class="i" props={{ innerHTML: "&#xf019;" }} /><span class="ptitle">Not downloaded</span>
        <span class="hint">the models a card offers that are not here · a request for one answers model_not_downloaded, unless:</span>
        <label class="hint" attrs={{ title: "NS_AUTO_PULL: a request for one of these starts its download and waits for it (a job says downloading <model>: N%) - no restart needed" }}>
          <input attrs={{ type: "checkbox" }} props={{ checked: st.auto_pull }} on={{ change: (e: Event) => void autoPull((e.target as HTMLInputElement).checked) }} />
          {" "}download automatically when a request needs one
        </label>
      </div>
      <div class="pbody">
        {kinds.map((k) => (
          <div class="kind">
            <h4 style={{ color: color(k) }}>{KIND_TITLE[k] ?? k}<span class="hint"> {list.filter((a) => a.kind === k).length} to download</span></h4>
            {list.filter((a) => a.kind === k).map(AvailableRow)}
          </div>
        ))}
      </div>
    </section>
  );
}

function ServiceRow(s: Service) {
  return (
    <div class="svc">
      <span class={{ dot: true, off: !s.up }} />
      <div><div class="name">{s.title}</div><div class="sub">{s.what}</div></div>
      <div class="state">{s.up ? <span>{s.detail ?? "up"}<span class="hint"> :{s.port}</span></span> : <span class="hint">not running · :{s.port}</span>}</div>
      <div class="acts">{s.up && s.link ? <a class="btn" attrs={{ href: s.href ?? at(s.port, s.link), target: "_blank" }}><span class="i" props={{ innerHTML: "&#xf08e;" }} /> open</a> : null}</div>
    </div>
  );
}

function Runs() {
  const runs = ui.state?.runs ?? [];
  if (!runs.length) return null;
  return (
    <section class="panel">
      <div class="phead"><span class="i" props={{ innerHTML: "&#xf120;" }} /><span class="ptitle">Actions</span><span class="hint">newest first</span></div>
      <div class="pbody">
        <ul class="runs">
          {runs.map((r) => (
            <li>
              <a attrs={{ href: "#" }} on={{ click: (e: Event) => { e.preventDefault(); ui.openRun = ui.openRun === r.id ? null : r.id; render(); } }}>{r.label}</a>{" "}
              <span class={r.rc === null ? "warn" : r.rc === 0 ? "ok" : "bad"}>{r.rc === null ? "running…" : r.rc === 0 ? "done" : "failed"}</span>
              <span class="hint"> {new Date(r.started * 1000).toLocaleTimeString()}</span>
              {ui.openRun === r.id || r.rc === null || (r.rc !== 0 && ui.openRun === null) ? <pre>{r.out || "…"}</pre> : null}
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}

function Status() {
  const st = ui.state;
  return (
    <div>
      <section class="panel">
        <div class="phead"><span class="i" props={{ innerHTML: "&#xf2db;" }} /><span class="ptitle">GPUs</span>
          <span class="hint">in use, by process · below it, hatched: what the enabled models are vouched for - each model's weights, then its headroom (the stripes the other way)</span></div>
        <div class="pbody"><div class="cards">{(st?.cards ?? []).map(CardView)}</div></div>
      </section>
      <DrivesPanel />
      <Models />
      <NotDownloaded />
      <section class="panel">
        <div class="phead"><span class="i" props={{ innerHTML: "&#xf233;" }} /><span class="ptitle">Front ends</span></div>
        <div class="pbody">{(st?.services ?? []).map(ServiceRow)}</div>
      </section>
      <Runs />
    </div>
  );
}

export function App() {
  const h = ui.state?.host;
  const mt = ui.state?.maintenance ?? null;
  return (
    <div id="app">
      <div class="top">
        <h1 class="brand">
          <img attrs={{ src: "/static/logo.svg", alt: "nextsycl" }} />
          <span class="host">on {location.hostname}</span>
        </h1>
        {h
          ? <div class="pill">
              <span>RAM <b>{gib(h.mem_total_mb - h.mem_avail_mb)}</b> / {gib(h.mem_total_mb)} GiB</span>
              <span>load <b>{h.load1?.toFixed(1) ?? "?"}</b> / {h.cpus} CPUs</span>
            </div>
          : null}
        {ui.state
          ? <button class={{ btn: true, maint: true, on: !!mt }} attrs={{ title: mt ? "end maintenance: queued work starts again" : "set the box aside for development: callers told, new work refused" }}
              on={{ click: () => void maintenance(!mt) }}>{mt ? "End maintenance" : "Maintenance"}</button>
          : null}
      </div>
      {mt
        ? <div class="maintbar">
            <b>Maintenance</b> {mt.notice}
            {mt.bypass
              ? <span class="hint"> · bypass header <code>X-Maintenance-Bypass: {mt.bypass}</code>{" "}
                  <button class="btn small" on={{ click: () => void copy(mt.bypass ?? "", "bypass") }}>{ui.copied === "bypass" ? "copied" : "copy"}</button></span>
              : null}
          </div>
        : null}
      <div class="tabs">
        <button class={{ tab: true, on: ui.tab === "status" }} on={{ click: () => setTab("status") }}>Status</button>
        <button class={{ tab: true, on: ui.tab === "api" }} on={{ click: () => setTab("api") }}>API</button>
        <button class={{ tab: true, on: ui.tab === "config" }} on={{ click: () => setTab("config") }}>Config</button>
      </div>
      {ui.error ? <pre class="err" on={{ click: () => { ui.error = null; render(); } }}>{ui.error}</pre> : null}
      {ui.tab === "api" ? <Api /> : ui.tab === "config" ? <ConfigView /> : <Status />}
    </div>
  );
}
