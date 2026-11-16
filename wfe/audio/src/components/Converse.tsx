/** The live conversation (a full-duplex speech model: info.realtime): start and stop, the voice (the model's own or a
 *  saved one, cloned), the language, a system prompt - or for a lockstep model (Haru-4B) its character card; while it
 *  runs the microphone's level, what is happening (listening, you are speaking, thinking, speaking) and the running
 *  transcript - the model hears your speech itself and writes none of it down, so your turns show as their length;
 *  its answers as they are written. */
import { jsx } from "../jsx.js";
import { clearTranscript, lockstep, setCard, setLanguage, setVoice, start, stop, talk } from "../converse.js";
import { render, state } from "../state.js";

const pretty = (v: string) => v.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());

/** A lockstep model's character card: its fields (empty ones keep the model's own character) */
function Card() {
  const field = (k: string, label: string, ph: string) => (
    <label class="cardf">
      <span class="lab">{label}</span>
      <input attrs={{ placeholder: ph, disabled: talk.on }} props={{ value: talk.card[k] ?? "" }}
             on={{ input: (e: Event) => setCard(k, (e.target as HTMLInputElement).value) }} />
    </label>
  );
  const pick = (k: string, label: string, opts: [string, string][]) => (
    <label class="cardf">
      <span class="lab">{label}</span>
      <select attrs={{ disabled: talk.on }} on={{ change: (e: Event) => setCard(k, (e.target as HTMLSelectElement).value) }}>
        {opts.map(([v, t]) => <option attrs={{ value: v }} props={{ selected: (talk.card[k] ?? "") === v }}>{t}</option>)}
      </select>
    </label>
  );
  return (
    <div>
      <div class="lab">character card <span class="hint">who it is (set before starting; empty fields keep its own: Haru, a college senior meeting you)</span></div>
      <textarea class="small" attrs={{ placeholder: "persona: Haru is a 29-year-old nurse in Denver, warm, chatty, laughs easily.", disabled: talk.on }}
                props={{ value: talk.card.persona ?? "" }} on={{ input: (e: Event) => setCard("persona", (e.target as HTMLTextAreaElement).value) }} />
      <div class="row">
        {field("name", "its name", "Haru")}
        {field("other_name", "your name", "Sam")}
        {pick("gender", "gender", [["", "-"], ["female", "female"], ["male", "male"]])}
        {pick("backchannel", "back-channels", [["", "-"], ["low", "sparse"], ["mid", "moderate"], ["high", "frequent"]])}
      </div>
      <div class="row">
        {field("relationship", "relationship", "Old friends from high school.")}
        {field("situation", "situation", "A phone call to catch up after a long time.")}
      </div>
    </div>
  );
}

export function Converse() {
  const rt = state.info?.realtime;
  if (!rt) return null;
  const saved = rt.clone ? state.voices.filter((v) => v.kind === "clone" || v.kind === "design") : [];
  // a model without voices of its own (Haru) speaks a saved one: the first, until another is picked
  const voice = talk.voice || rt.voices[0] || saved[0]?.name || "";
  const level = Math.round(talk.level * 100);
  const ls = lockstep();
  return (
    <div class="create converse">
      <div class="row">
        {talk.on || talk.connecting
          ? <button type="button" class="go stop" on={{ click: () => stop() }}>{talk.connecting ? "connecting…" : "■ Stop"}</button>
          : <button type="button" class="go" on={{ click: () => { if (!talk.voice && voice) talk.voice = voice; void start(); } }}>● Start talking</button>}
        <span class="hint">{talk.on ? talk.phase : ls
          ? "full duplex: it listens while it speaks, a frame every 133 ms - wired headphones keep it from hearing itself"
          : "the microphone streams to the model; it answers when you have finished, and stops when you cut in"}</span>
        {talk.on
          ? <span class="meter" attrs={{ title: "your microphone" }}><span class="bar" style={{ width: `${level}%` }} /></span>
          : null}
        {talk.latency != null && !ls ? <span class="hint">{`first audio ${talk.latency.toFixed(2)} s after you stopped`}</span> : null}
      </div>
      <div>
        <div class="lab">voice</div>
        <span class="chips">
          {rt.voices.length === 0 && saved.length === 0
            ? <span class="hint">it speaks in a voice it is given: save one first (Voices: 3 s to 3 minutes of one speaker)</span>
            : null}
          {rt.voices.map((v) => (
            <button type="button" class={{ chip: true, on: voice === v }} on={{ click: () => setVoice(v) }}>{pretty(v)}</button>
          ))}
          {saved.map((v) => (
            <button type="button" class={{ chip: true, on: voice === v.name }} attrs={{ title: `a saved voice (${v.seconds.toFixed(1)} s), cloned` }}
                    on={{ click: () => setVoice(v.name) }}>{`★ ${v.name}`}</button>
          ))}
        </span>
      </div>
      <div class="row">
        <div class="lab">language</div>
        <select props={{ value: talk.language }} on={{ change: (e: Event) => setLanguage((e.target as HTMLSelectElement).value) }}>
          {rt.languages.map((l) => <option attrs={{ value: l }} props={{ selected: talk.language === l }}>{l}</option>)}
        </select>
      </div>
      {ls ? <Card /> : <div class="lab">instructions <span class="hint">optional: the model's own system prompt when empty (set before starting)</span></div>}
      {ls ? null : <textarea class="small" attrs={{ placeholder: "You are a friendly barista; keep answers short.", disabled: talk.on }} props={{ value: talk.instructions }}
                on={{ input: (e: Event) => { talk.instructions = (e.target as HTMLTextAreaElement).value; render(); } }} />}
      {talk.error ? <pre class="err">{talk.error}</pre> : null}
      <div class="row">
        <div class="lab">transcript</div>
        {talk.lines.length ? <button type="button" class="tbtn" on={{ click: () => clearTranscript() }}>clear</button> : null}
      </div>
      <div class="transcript">
        {talk.lines.length === 0 ? <div class="hint">nothing said yet</div> : null}
        {talk.lines.map((l) => (
          <div class={{ turn: true, [l.role]: true, live: l.status === "live", cut: l.status === "cut" }}>
            <span class="who">{l.role === "user" ? "you" : l.role === "assistant" ? (state.info?.model ?? "model") : l.role}</span>
            <span class="said">
              {l.role === "user" ? (l.status === "live" ? "🎙 speaking…" : "🎙 (heard)") : l.text || (l.status === "live" ? "…" : "")}
              {l.status === "cut" ? <span class="hint"> (cut short)</span> : null}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
