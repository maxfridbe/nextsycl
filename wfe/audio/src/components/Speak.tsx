/** The speech request: the text, the voice (a built-in one, one made up from a description, or one cloned from a
 *  recording - whichever the model does), the language, the most to make, the seed; under Advanced the engine's own
 *  options. Each field writes straight into state.form. */
import { jsx } from "../jsx.js";
import { canGo, cancel, deleteVoice, enrollJob, enrollVoice, generate, pickReference, randomSeed, saveVoice } from "../actions.js";
import { panelOpen, render, saveForm, state, togglePanel } from "../state.js";

type Text = "text" | "instructions" | "refText" | "voice" | "language";

function set(k: Text, v: string): void {
  state.form[k] = v;
  saveForm();
  render();
}

const area = (k: "text" | "instructions" | "refText", cls: string, placeholder: string) => (
  <textarea
    class={cls}
    attrs={{ placeholder }}
    props={{ value: state.form[k] }}
    on={{
      input: (e: Event) => { state.form[k] = (e.target as HTMLTextAreaElement).value; saveForm(); render(); },
      keydown: (e: KeyboardEvent) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) void generate(); },
    }}
  />
);

const pretty = (v: string) => v.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());

/** A saved voice's enrolment for the model without an encoder (Voxtral TTS, through the front door): its job's
 *  progress, done, or the button that starts one; nothing when the page is not opened through the door */
function Enroll(name: string, enrolled: string[] | undefined) {
  const m = state.enroll?.models[0];
  if (!m) return null;
  const who = m.title?.split(" ")[0] ?? m.id;
  const j = enrollJob(name);
  if (j && (j.state === "queued" || j.state === "running")) {
    const p = j.progress;
    const what = p?.at && p.of ? `fitting ${Math.round((100 * p.at) / p.of)}%` : j.waiting_for ?? j.stage ?? "queued";
    return <span class="hint" attrs={{ title: j.waiting_for ?? "" }}>{`enrolling for ${who}: ${what}`}</span>;
  }
  if (enrolled?.includes(m.arch)) return <span class="hint" attrs={{ title: `${name} is enrolled for ${m.id}` }}>{`✓ ${who}`}</span>;
  return (
    <button type="button" class="tbtn" attrs={{ title: j?.state === "failed" ? `the last try failed: ${j.error?.message ?? ""}` : `fit ${name} for ${m.id} (experimental; a minute or two on a GPU)` }}
            on={{ click: () => void enrollVoice(name) }}>{j?.state === "failed" ? `enroll for ${who} (failed - again)` : `enroll for ${who}`}</button>
  );
}

export function Speak() {
  const f = state.form;
  const sp = state.info?.speech;
  if (!sp) return <div class="hint">connecting…</div>;
  const max = state.info?.defaults?.max_seconds ?? 600;
  const open = panelOpen("advanced");
  return (
    <div class="create">
      <div class="lab">text <span class="hint">what to say; punctuation sets the pauses</span></div>
      {area("text", "", "Good morning. The bread is still warm, and the coffee is ready.")}
      {sp.events?.length
        ? <div class="row">
            <span class="hint">vocal events, where they happen in the text:</span>
            <span class="chips">
              {sp.events.map((e) => (
                <button type="button" class="chip" attrs={{ title: `add ${e} to the text` }}
                        on={{ click: () => set("text", `${f.text}${f.text && !f.text.endsWith(" ") ? " " : ""}${e} `) }}>{e}</button>
              ))}
            </span>
          </div>
        : null}
      {sp.voices.length
        ? <div>
            <div class="lab">voice</div>
            <span class="chips">
              {sp.voices.map((v) => (
                <button type="button" class={{ chip: true, on: f.voice === v }} on={{ click: () => set("voice", v) }}>{pretty(v)}</button>
              ))}
            </span>
          </div>
        : null}
      {sp.design || sp.instructions
        ? <div>
            <div class="lab">{sp.requires === "instructions" ? "the voice" : sp.design ? "the voice, or the style" : "how to say it"}
              <span class="hint">{sp.requires === "instructions" ? "who speaks: age, gender, timbre, pace, emotion, accent"
                : sp.design && sp.direction ? "optional: a voice to make up (who speaks: age, gender, timbre, pace; in the language of the text), or with a cloned voice its direction: tone, emotion, pace, delivery"
                : sp.design ? "optional: a voice to make up (who speaks: age, gender, timbre, pace), or with a cloned voice its style"
                : "optional: the mood, the pace, the delivery"}</span>
            </div>
            {area("instructions", "", sp.design
              ? "A calm elderly woman with a soft, slightly raspy voice, speaking slowly and warmly."
              : "Cheerful and a little hurried, as if late for a train.")}
          </div>
        : null}
      {!sp.clone && sp.enrolled && state.voices.length
        ? <div>
            <div class="lab">a saved voice <span class="hint">experimental: spoken once enrolled for this model (fitted on a GPU, a minute or two)</span></div>
            <span class="chips">
              {state.voices.map((v) => (
                <span class="vchip">
                  {v.enrolled?.includes(state.info?.arch ?? "")
                    ? <button type="button" class={{ chip: true, on: f.voice === v.name }} attrs={{ title: v.description ?? v.text ?? "" }}
                              on={{ click: () => set("voice", f.voice === v.name ? "" : v.name) }}>{v.name}</button>
                    : <span class="chip off" attrs={{ title: "not enrolled for this model yet" }}>{v.name}</span>}
                  {v.enrolled?.includes(state.info?.arch ?? "") ? null : Enroll(v.name, v.enrolled)}
                </span>
              ))}
            </span>
            {state.enroll ? null : <div class="hint">open this page through the front door (:8000) to enroll a voice</div>}
          </div>
        : null}
      {sp.clone
        ? <div>
            {state.voices.length
              ? <div>
                  <div class="lab">a saved voice <span class="hint">{sp.requires === "reference" ? "or a recording below" : "optional: or a recording below"}</span></div>
                  <span class="chips">
                    {state.voices.map((v) => (
                      <span class="vchip">
                        <button type="button" class={{ chip: true, on: f.voice === v.name }} attrs={{ title: v.description ?? v.text ?? "" }}
                                on={{ click: () => set("voice", f.voice === v.name ? "" : v.name) }}>{v.name}{v.kind === "design" ? " ✦" : ""}</button>
                        <button type="button" class="tbtn x" attrs={{ title: `delete ${v.name}` }} on={{ click: () => void deleteVoice(v.name) }}>×</button>
                        {f.voice === v.name ? Enroll(v.name, v.enrolled) : null}
                      </span>
                    ))}
                  </span>
                </div>
              : null}
            {state.voices.some((v) => v.name === f.voice)
              ? <div class="row"><audio attrs={{ src: `v1/audio/voices/${f.voice}.wav`, controls: true, preload: "none" }} /></div>
              : <div>
                  <div class="lab">the voice to clone <span class="hint">3-15 s of one person speaking clearly (WAV, MP3, FLAC, Ogg, M4A)</span></div>
                  <div class="row">
                    <input attrs={{ type: "file", accept: "audio/*,.wav,.mp3,.flac,.ogg,.m4a" }}
                           on={{ change: (e: Event) => pickReference((e.target as HTMLInputElement).files?.[0] ?? null) }} />
                    {f.refAudio ? <audio attrs={{ src: f.refAudio, controls: true, preload: "auto" }} /> : null}
                    {f.refAudio ? <span class="hint">{f.refName}</span> : null}
                    {f.refAudio
                      ? <button type="button" class="tbtn" on={{ click: () => void saveVoice({ sample: f.refAudio, text: f.refText }) }}>save as voice…</button>
                      : null}
                  </div>
                  {sp.clone_needs_text || sp.clone_takes_text
                    ? <div>
                        <div class="lab">what the recording says
                          <span class="hint">{sp.clone_needs_text ? "required" : "optional: with it the clone continues the recording itself - a closer voice"}</span>
                        </div>
                        {area("refText", "", "Good morning. The bread is still warm and the coffee is ready.")}
                      </div>
                    : null}
                </div>}
          </div>
        : null}
      <div class="row">
        {sp.languages.length
          ? <label>language
              <select on={{ change: (e: Event) => set("language", (e.target as HTMLSelectElement).value) }}>
                <option attrs={{ value: "" }} props={{ selected: f.language === "" }}>auto</option>
                {sp.languages.map((l) => <option attrs={{ value: l }} props={{ selected: f.language === l }}>{pretty(l)}</option>)}
              </select>
            </label>
          : null}
        <label>at most (s)
          <input attrs={{ type: "number", min: 1, max, step: 1 }} props={{ value: String(f.seconds) }}
                 on={{ change: (e: Event) => { const x = Number((e.target as HTMLInputElement).value); if (x > 0) { state.form.seconds = Math.min(x, max); saveForm(); } render(); } }} />
        </label>
        <label>seed
          <span class="seed">
            <input attrs={{ type: "number", min: 0, placeholder: "random" }} props={{ value: f.seed === null ? "" : String(f.seed) }}
                   on={{ change: (e: Event) => { const t = (e.target as HTMLInputElement).value.trim(); state.form.seed = t === "" ? null : Number(t); saveForm(); render(); } }} />
            <button type="button" class="tbtn" attrs={{ title: "a new random seed" }} on={{ click: randomSeed }}>
              <span class="i" props={{ innerHTML: "&#xf074;" }} />
            </button>
          </span>
        </label>
      </div>
      <div class="sub">
        <div class="subhead" on={{ click: () => togglePanel("advanced") }}><span class="chev">{open ? "▾" : "▸"}</span> Advanced</div>
        {open
          ? <div class="subbody">
              <div class="row">
                {sp.design || sp.direction
                  ? <label attrs={{ title: "guidance with an instruction (1: none)" }}>cfg
                      <input attrs={{ type: "number", min: 1, max: 10, step: 0.5, placeholder: String(state.info?.defaults?.cfg ?? "") }}
                             props={{ value: f.cfg === null ? "" : String(f.cfg) }}
                             on={{ change: (e: Event) => { const t = (e.target as HTMLInputElement).value.trim(); state.form.cfg = t === "" ? null : Number(t); saveForm(); render(); } }} />
                    </label>
                  : null}
                {(state.info?.options ?? []).map((o) => (
                  <label attrs={{ title: o.help }}>{o.name}
                    <input attrs={{ type: "text", placeholder: o.value || "on" }} props={{ value: f.options[o.name] ?? "" }}
                           on={{ change: (e: Event) => { state.form.options[o.name] = (e.target as HTMLInputElement).value; saveForm(); render(); } }} />
                  </label>
                ))}
              </div>
              <div class="hint">The talker draws a frame 12.5 times a second of speech, its code predictor the other fifteen codes; the codec turns them into sound.</div>
            </div>
          : null}
      </div>
      <div class="row">
        <button type="button" attrs={{ id: "go", disabled: state.busy || !canGo() }} on={{ click: () => void generate() }}>
          <span class="i" props={{ innerHTML: "&#xf130;" }} />{state.busy ? "Speaking…" : "Speak"}
        </button>
        {state.busy || state.progress?.busy
          ? <button type="button" attrs={{ id: "stop" }} on={{ click: () => void cancel() }}><span class="i" props={{ innerHTML: "&#xf04d;" }} />Cancel</button>
          : null}
      </div>
    </div>
  );
}
