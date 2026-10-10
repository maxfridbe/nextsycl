/** The page: the host strip and the tabs - Status (the GPUs with what holds them and what is vouched for them, the
 *  models by kind with enable / start / stop, the front ends, the actions' output) and API. */
import { jsx } from "../jsx.js";
import { act, render, setTab, ui } from "../main.js";
import type { Card, Model, Proc, Service } from "../types.js";
import { Api } from "./Api.js";
import { ConfigView } from "./Config.js";

const gib = (mb: number) => (mb / 1024).toFixed(1);
const short = (n: string) => n.replace("Intel(R) Arc(TM) ", "").replace(" Graphics", "");
/** a link to a service on this box, as the browser reaches it */
const at = (port: number, path = "/") => `${location.protocol}//${location.hostname}:${port}${path}`;

const COLORS: Record<string, string> = { llm: "#06b6d4", image: "#f59e0b", video: "#ec4899", audio: "#8b5cf6" };
const color = (k?: string) => COLORS[k ?? ""] ?? "#64748b";
const KINDS: { kind: Model["kind"]; title: string; icon: string }[] = [
  { kind: "llm", title: "Chat", icon: "&#xf086;" },
  { kind: "image", title: "Images", icon: "&#xf03e;" },
  { kind: "audio", title: "Music", icon: "&#xf001;" },
  { kind: "video", title: "Video", icon: "&#xf03d;" },
];

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
        {c.vouched.map((v) => <i class={{ hatch: true, live: ["loaded", "busy"].includes(v.state) }}
          style={{ width: `${Math.min(100, (v.mb / total) * 100)}%`, "--c": color(v.kind) } as Record<string, string>} />)}
      </div>
      <div class="kv">
        <span>in use <b>{gib(c.vram_used_mb)}</b> / {gib(c.vram_total_mb)} GiB · <span class={free < 1536 ? "bad" : ""}>{gib(free)} free</span></span>
        <span class={c.overbooked ? "warn" : ""}>vouched <b>{gib(c.vouched_mb)}</b> GiB{c.overbooked ? " - overbooked: the first to load holds the card" : ""}</span>
        <span>busy <b>{Math.round(c.busy_pct)}%</b></span>
        {c.power_w != null ? <span>power <b>{Math.round(c.power_w)} W</b>{c.power_cap_w ? ` / ${c.power_cap_w}` : ""}</span> : null}
        {c.temp_pkg != null ? <span>temp <b>{Math.round(c.temp_pkg)}°C</b>{c.temp_vram != null ? ` · mem ${Math.round(c.temp_vram)}°C` : ""}</span> : null}
        {c.pcie?.cur ? <span>PCIe <b>{c.pcie.cur.gen ? `${c.pcie.cur.gen}.0` : "?"} x{c.pcie.cur.width}</b></span> : null}
      </div>
      <ul class="procs">
        {c.procs.map((p) => (
          <li attrs={{ title: p.args ?? `pid ${p.pid}` }}>
            <span class="sw" style={{ background: color(p.kind) }} />
            <span class="what">{procName(p)}{p.port ? <span class="hint"> :{p.port}</span> : null}</span>
            <span class="mb">{gib(p.vram_mb)} GiB</span>
          </li>
        ))}
        {c.vouched.filter((v) => !["loaded", "busy"].includes(v.state)).map((v) => (
          <li class="dim">
            <span class="sw hatch" style={{ "--c": color(v.kind) } as Record<string, string>} />
            <span class="what">{v.model}{v.of && v.of > 1 ? <span class="hint"> (one of {v.of} {v.kind === "llm" ? "chat models" : "video engines"}: one runs at a time)</span> : null}
              <span class="hint"> {v.state}{v.measured ? "" : " · estimate"}</span></span>
            <span class="mb">{gib(v.mb)} GiB</span>
          </li>
        ))}
        {!c.procs.length && !c.vouched.length ? <li class="dim"><span class="what hint">free, nothing enabled here</span></li> : null}
      </ul>
    </div>
  );
}

const STATE_TEXT: Record<string, string> = {
  disabled: "disabled", unloaded: "enabled · unloaded", loading: "loading…", loaded: "loaded", busy: "working", ready: "ready",
};

function GpuPicker(m: Model) {
  const cards = ui.state?.cards ?? [];
  const pick = (ui.pick[m.id] ??= [...m.gpus]);
  const single = m.kind === "image" || m.kind === "audio";
  return (
    <span class="gpus" attrs={{ title: single ? "the card it loads on" : "the cards it may use" }}>
      {cards.map((c) => (
        <label class={{ gpu: true, on: pick.includes(c.index) }}>
          <input attrs={{ type: single ? "radio" : "checkbox", name: `g-${m.id}`, checked: pick.includes(c.index), disabled: m.enabled }}
            on={{ change: (e: Event) => {
              const on = (e.target as HTMLInputElement).checked;
              ui.pick[m.id] = single ? [c.index] : on ? [...new Set([...pick, c.index])].sort() : pick.filter((g) => g !== c.index);
              render();
            } }} />
          {short(c.name).replace("Pro ", "")}
        </label>
      ))}
    </span>
  );
}

function ModelRow(m: Model) {
  const p = ui.pending[m.id];
  const spin = (what: string) => p?.what === what || (what === "load" && m.state === "loading");
  const loaded = ["loaded", "busy", "ready"].includes(m.state);
  const btn = (what: string, label: string, cls: string, title: string, extra: Record<string, unknown> = {}) => (
    <button class={cls} attrs={{ disabled: !!p || (what !== "load" && m.state === "loading"), title }}
      on={{ click: () => void act(what, m.id, extra) }}>
      {spin(what) ? <span class="spin" /> : null}{label}
    </button>
  );
  const vouch = Object.entries(m.vouch_mb).map(([g, mb]) => `GPU ${g}: ${gib(mb)} GiB`).join(", ");
  const page = m.kind === "llm" ? at(m.page) : at(m.page);
  return (
    <div class={{ mrow: true, off: !m.enabled }}>
      <span class={{ dot: true, off: !loaded, busy: m.state === "busy" || m.state === "loading" }} />
      <div class="mname">
        <div class="name">{m.title ?? m.id}</div>
        <div class="sub">{m.id} · <span attrs={{ title: m.measured ? "as seen while it ran" : "an estimate from its files, until it has run" }}>{vouch}{m.measured ? "" : " (est.)"}</span></div>
      </div>
      <div class="mstate"><span class={{ tag: true, [m.state]: true }}>{STATE_TEXT[m.state] ?? m.state}</span></div>
      <div class="acts">
        {GpuPicker(m)}
        {m.enabled
          ? [
              loaded
                ? btn("unload", "stop", "ghost", "unload now (after the requests it is answering); it stays enabled")
                : btn("load", m.state === "loading" ? "starting" : "start", "", "load now (otherwise its first request does); the button spins until it answers"),
              btn("disable", "disable", "ghost danger", m.kind === "llm" ? "unload it and take it off Open WebUI's list" : "unload it and take it off the API",
                  m.kind === "video" ? { force: false } : {}),
              loaded || m.kind === "video" || m.kind === "llm"
                ? <a class="btn" attrs={{ href: page, target: "_blank", title: m.kind === "llm" ? "Open WebUI" : "its page" }}><span class="i" props={{ innerHTML: "&#xf08e;" }} /></a>
                : null,
            ]
          : btn("enable", "enable", "", "offer it on the cards picked: it loads on its first request, unloads after the idle time",
                { gpus: ui.pick[m.id] ?? m.gpus })}
      </div>
    </div>
  );
}

function Models() {
  const st = ui.state;
  if (!st) return null;
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

function ServiceRow(s: Service) {
  return (
    <div class="svc">
      <span class={{ dot: true, off: !s.up }} />
      <div><div class="name">{s.title}</div><div class="sub">{s.what}</div></div>
      <div class="state">{s.up ? <span>{s.detail ?? "up"}<span class="hint"> :{s.port}</span></span> : <span class="hint">not running · :{s.port}</span>}</div>
      <div class="acts">{s.up && s.link ? <a class="btn" attrs={{ href: at(s.port, s.link), target: "_blank" }}><span class="i" props={{ innerHTML: "&#xf08e;" }} /> open</a> : null}</div>
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
          <span class="hint">in use, by process · below it, hatched: what the enabled models are vouched for</span></div>
        <div class="pbody"><div class="cards">{(st?.cards ?? []).map(CardView)}</div></div>
      </section>
      <Models />
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
      </div>
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
