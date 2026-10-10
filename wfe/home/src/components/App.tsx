/** The page: the host strip, a card per GPU with what holds it, the services with their links and controls, and the
 *  actions' output. */
import { jsx } from "../jsx.js";
import { act, render, ui } from "../main.js";
import type { Card, Proc, Service } from "../types.js";

const gib = (mb: number) => (mb / 1024).toFixed(1);
const short = (n: string) => n.replace("Intel(R) Arc(TM) ", "").replace(" Graphics", "");
/** a link to a service on this box, as the browser reaches it */
const at = (port: number, path = "/") => `${location.protocol}//${location.hostname}:${port}${path}`;

const COLORS: Record<string, string> = { llm: "#3b82f6", image: "#a855f7", video: "#f59e0b", audio: "#22c55e" };
const color = (p: Proc) => COLORS[p.kind ?? ""] ?? "#64748b";

function procName(p: Proc): string {
  if (p.program !== "nextsycl") return p.program;
  return [p.kind, p.command, p.model].filter((x) => x).join(" ");
}

function CardView(c: Card) {
  const total = Math.max(c.vram_total_mb, 1);
  const used = c.vram_used_mb;
  const free = c.vram_total_mb - used;
  return (
    <div class="card">
      <h3>
        <span class={{ dot: true, busy: c.busy_pct > 5, off: used < 512 }} />
        {short(c.name)} <span class="idx">GPU {c.index} · {c.pci}</span>
      </h3>
      <div class="meter" attrs={{ title: `${gib(used)} of ${gib(c.vram_total_mb)} GiB` }}>
        {c.procs.map((p) => <i style={{ width: `${(p.vram_mb / total) * 100}%`, background: color(p) }} />)}
      </div>
      <div class="kv">
        <span>VRAM <b>{gib(used)}</b> / {gib(c.vram_total_mb)} GiB · <span class={free < 1536 ? "bad" : ""}>{gib(free)} free</span></span>
        <span>busy <b>{Math.round(c.busy_pct)}%</b></span>
        {c.power_w != null ? <span>power <b>{Math.round(c.power_w)} W</b>{c.power_cap_w ? ` / ${c.power_cap_w}` : ""}</span> : null}
        {c.temp_pkg != null ? <span>temp <b>{Math.round(c.temp_pkg)}°C</b>{c.temp_vram != null ? ` · mem ${Math.round(c.temp_vram)}°C` : ""}</span> : null}
        {c.pcie?.cur ? <span>PCIe <b>{c.pcie.cur.gen ? `${c.pcie.cur.gen}.0` : "?"} x{c.pcie.cur.width}</b></span> : null}
      </div>
      {c.procs.length
        ? <ul class="procs">
            {c.procs.map((p) => (
              <li attrs={{ title: p.args ?? `pid ${p.pid}` }}>
                <span class="sw" style={{ background: color(p) }} />
                <span class="what">{procName(p)}{p.port ? <span class="hint"> :{p.port}</span> : null}</span>
                <span class="mb">{gib(p.vram_mb)} GiB</span>
              </li>
            ))}
          </ul>
        : <div class="hint">free</div>}
    </div>
  );
}

function ChatPicker(s: Service) {
  const m = s.chat;
  if (!m?.choices) return null;
  return (
    <select
      attrs={{ disabled: ui.sending || !!m.starting, title: "the chat model (the video studio swaps it, waiting for a render that holds the card)" }}
      on={{ change: (e: Event) => void act({ do: "mode", model: (e.target as HTMLSelectElement).value }) }}
    >
      <option attrs={{ value: "none", selected: !m.mode || m.mode === "none" }}>no chat model (card free)</option>
      {Object.entries(m.choices).map(([id, title]) => (
        <option attrs={{ value: id, selected: m.mode === id }}>{title.replace(/ \(nextsycl\)$/, "")}</option>
      ))}
    </select>
  );
}

function StartControls(s: Service) {
  const st = ui.state;
  const kind = s.kind ?? s.id;
  const models = (st?.models ?? []).filter((m) => m.kind === kind);
  if (!st || !models.length) return [<span class="hint">no {kind} model registered</span>];
  const pick = (ui.pick[kind] ??= { model: models[0]?.id ?? "", gpu: freest(st.cards) });
  const conflict = ui.conflict?.kind === kind ? ui.conflict.message : null;
  return [
    <select on={{ change: (e: Event) => { pick.model = (e.target as HTMLSelectElement).value; render(); } }}>
      {models.map((m) => <option attrs={{ value: m.id, selected: pick.model === m.id }}>{m.id}</option>)}
    </select>,
    <select on={{ change: (e: Event) => { pick.gpu = Number((e.target as HTMLSelectElement).value); render(); } }}>
      {st.cards.map((c) => (
        <option attrs={{ value: String(c.index), selected: pick.gpu === c.index }}>
          GPU {c.index} {short(c.name)} ({gib(c.vram_total_mb - c.vram_used_mb)} GiB free)
        </option>
      ))}
    </select>,
    conflict
      ? <button class="danger" attrs={{ disabled: ui.sending, title: conflict }}
          on={{ click: () => void act({ do: "start", kind, model: pick.model, gpu: pick.gpu, force: true }) }}>start anyway</button>
      : <button attrs={{ disabled: ui.sending }} on={{ click: () => void act({ do: "start", kind, model: pick.model, gpu: pick.gpu }) }}>
          <span class="i" props={{ innerHTML: "&#xf04b;" }} />start
        </button>,
    conflict ? <div class="hint warn" style={{ flexBasis: "100%", textAlign: "right" }}>{conflict}</div> : null,
  ];
}

/** the card with the most free VRAM */
function freest(cards: Card[]): number {
  let best = cards[0];
  for (const c of cards) if (best && c.vram_total_mb - c.vram_used_mb > best.vram_total_mb - best.vram_used_mb) best = c;
  return best?.index ?? 0;
}

function ServiceRow(s: Service) {
  const state = !s.up
    ? <span class="hint">not running · :{s.port}</span>
    : <span>
        {s.model ? <b>{s.model}</b> : null}
        {s.model && (s.detail || s.busy != null) ? " · " : ""}
        {s.detail ?? null}
        {s.busy != null ? <span class={s.busy ? "warn" : "ok"}>{s.busy ? "busy" : "idle"}{s.waiting ? `, ${s.waiting} waiting` : ""}</span> : null}
        <span class="hint"> :{s.port}{s.api ? ` · API ${s.api}` : ""}</span>
      </span>;
  const acts = [
    s.up && s.link && s.page !== false ? <a class="btn" attrs={{ href: at(s.port, s.link), target: "_blank" }}><span class="i" props={{ innerHTML: "&#xf08e;" }} /> open</a> : null,
    s.id === "video" && s.up ? ChatPicker(s) : null,
    ...(s.startable && !s.up ? StartControls(s) : []),
    s.startable && s.up
      ? <button class="ghost" attrs={{ disabled: ui.sending, title: "waits for the requests in progress" }}
          on={{ click: () => void act({ do: "stop", kind: s.kind }) }}>stop</button>
      : null,
  ];
  return (
    <div class="svc">
      <span class={{ dot: true, off: !s.up, busy: !!s.busy }} />
      <div><div class="name">{s.title}</div><div class="sub">{s.what}</div></div>
      <div class="state">{state}</div>
      <div class="acts">{acts}</div>
    </div>
  );
}

function Runs() {
  const runs = ui.state?.runs ?? [];
  if (!runs.length) return null;
  return (
    <section class="panel">
      <div class="phead"><span class="i" props={{ innerHTML: "&#xf120;" }} /><span class="ptitle">Actions</span><span class="hint">from this page, newest first</span></div>
      <div class="pbody">
        <ul class="runs">
          {runs.map((r) => (
            <li>
              <a attrs={{ href: "#" }} on={{ click: (e: Event) => { e.preventDefault(); ui.openRun = ui.openRun === r.id ? null : r.id; render(); } }}>
                {r.label}
              </a>{" "}
              <span class={r.rc === null ? "warn" : r.rc === 0 ? "ok" : "bad"}>{r.rc === null ? "running…" : r.rc === 0 ? "done" : `failed (${r.rc})`}</span>
              <span class="hint"> {new Date(r.started * 1000).toLocaleTimeString()}</span>
              {ui.openRun === r.id || r.rc === null || (r.rc !== 0 && ui.openRun === null) ? <pre>{r.out || "…"}</pre> : null}
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}

export function App() {
  const st = ui.state;
  const h = st?.host;
  return (
    <div id="app">
      <div class="top">
        <h1><span class="i" props={{ innerHTML: "&#xf108;" }} /> nextsycl on {location.hostname}</h1>
        {h
          ? <div class="pill">
              <span>RAM <b>{gib(h.mem_total_mb - h.mem_avail_mb)}</b> / {gib(h.mem_total_mb)} GiB</span>
              <span>load <b>{h.load1?.toFixed(1) ?? "?"}</b> / {h.cpus} CPUs</span>
            </div>
          : null}
      </div>
      {ui.error ? <pre class="err">{ui.error}</pre> : null}
      <section class="panel">
        <div class="phead"><span class="i" props={{ innerHTML: "&#xf2db;" }} /><span class="ptitle">GPUs</span>
          <span class="hint">VRAM by process (the DRM clients' fdinfo) · every 2 s</span></div>
        <div class="pbody"><div class="cards">{(st?.cards ?? []).map(CardView)}</div>
          {st && !st.cards.length ? <div class="hint">no Intel GPU on the xe driver</div> : null}</div>
      </section>
      <section class="panel">
        <div class="phead"><span class="i" props={{ innerHTML: "&#xf233;" }} /><span class="ptitle">Services</span>
          <span class="hint">what serves where; open a page, start or stop a server, pick the chat model</span></div>
        <div class="pbody">{(st?.services ?? []).map(ServiceRow)}</div>
      </section>
      <Runs />
    </div>
  );
}
