/** Everything to do with making a new clip. Deliberately separate from Jobs. */
import { jsx } from "../jsx.js";
import { rpc } from "../api.js";
import { render, state } from "../state.js";
import { navigate } from "../router.js";
import type { Engine, GenerateRequest } from "../types.js";
import { CanvasPicker } from "./CanvasPicker.js";
import { Panel } from "./Panel.js";

interface Form {
  prompt: string;
  seconds: number;
  steps: number;
  seed: number;
  te: "teacher" | "student";
  label: string;
  width: number;
  height: number;
  engine: Engine;
  upscale: number;
  /** "latent" (default), or an ESRGAN-type network on the decoded frames */
  upscaler: string;
  /** "" = the engine's own Euler / shifted schedule */
  sampler: string;
  schedule: string;
  chain: boolean;
  /** uploaded references (names in the studio's output directory) */
  refImages: string[];
  refVideos: string[];
  refAudios: string[];
  refVideoSound: boolean;
  refImageMax: boolean;
  controlVideo: string;
  controlMask: string;
  controlSource: string;
  controlStrength: number;
}

export const form: Form = {
  prompt:
    "integrated_multimodal_description: [Shot 1] Live-action, cinematic, a medium shot frames " +
    "a middle-aged baker with a calm, slightly raspy voice (S1) opening the shutters of a small " +
    "street bakery at dawn, and he says: <d>[English] Morning. The bread is still warm.</d>" +
    "\n\noverall_soundscape: Wooden shutters scrape open over a quiet street, trays clink inside." +
    "\n\nnon_diegetic_music: None.",
  seconds: 10,
  steps: 8,
  seed: 0,
  te: "teacher",
  label: "",
  width: 768,
  height: 576,
  engine: "INT8",
  upscale: 1.5,
  upscaler: "esrgan-general",
  sampler: "",
  schedule: "",
  chain: false,
  refImages: [],
  refVideos: [],
  refAudios: [],
  refVideoSound: true,
  refImageMax: false,
  controlVideo: "",
  controlMask: "",
  controlSource: "",
  controlStrength: 1,
};

/** Fill the form from an existing job - the "use these settings" path. */
export function useSettings(j: GenerateRequest): void {
  if (j.prompt && j.prompt !== "(recovered)") form.prompt = j.prompt;
  if (j.seconds) form.seconds = j.seconds;
  if (j.steps) form.steps = j.steps;
  if (j.seed != null) form.seed = j.seed;
  if (j.te) form.te = j.te;
  if (j.width) form.width = j.width;
  if (j.height) form.height = j.height;
  if (j.engine) form.engine = j.engine === "Q8_0" ? "INT8" : j.engine;
  if (j.upscale) form.upscale = j.upscale;
  form.upscaler = j.upscaler ?? "esrgan-general";
  form.sampler = j.sampler ?? "";
  form.schedule = j.schedule ?? "";
  form.refImages = j.ref_images ?? [];
  form.refVideos = j.ref_videos ?? [];
  form.refAudios = j.ref_audios ?? [];
  form.refVideoSound = j.ref_video_sound ?? true;
  form.refImageMax = j.ref_image_size === "max";
  form.controlVideo = j.control_video ?? "";
  form.controlMask = j.control_mask ?? "";
  form.controlSource = j.control_source ?? "";
  form.controlStrength = j.control_strength ?? 1;
  form.label = j.label ?? "";
  form.chain = !!j.first_frame;
  navigate({ tab: "create" });
}

async function generate(queue: boolean): Promise<void> {
  const body: GenerateRequest = {
    prompt: form.prompt,
    seconds: form.seconds,
    steps: form.steps,
    seed: form.seed,
    te: form.te,
    label: form.label,
    width: form.width,
    height: form.height,
    engine: form.engine,
    upscale: form.upscale,
    upscaler: form.upscaler,
    chain_mode: "png",
    queue,
  };
  if (form.sampler) body.sampler = form.sampler;
  if (form.schedule) body.schedule = form.schedule;
  if (form.refImages.length) body.ref_images = form.refImages;
  if (form.refVideos.length) {
    body.ref_videos = form.refVideos;
    body.ref_video_sound = form.refVideoSound;
  }
  if (form.refAudios.length) body.ref_audios = form.refAudios;
  if (form.refImages.length && form.refImageMax) body.ref_image_size = "max";
  if (form.controlVideo) body.control_video = form.controlVideo;
  if (form.controlMask) {
    body.control_mask = form.controlMask;
    if (form.controlSource) body.control_source = form.controlSource;
  }
  if (form.controlVideo || form.controlMask) body.control_strength = form.controlStrength;
  if (form.chain) body.first_frame = "prev";
  try {
    await rpc("generate", body);
    state.error = null;
  } catch (e) {
    state.error = String(e);
  }
  render();
}

const num = (
  label: string, key: "seconds" | "steps" | "seed" | "upscale",
  min: number, max: number, step: number,
) => (
  <label>
    {label}
    <input
      attrs={{ type: "number", min, max, step, value: String(form[key]) }}
      on={{ input: (e: Event) => { form[key] = Number((e.target as HTMLInputElement).value); render(); } }}
    />
  </label>
);

/** Comfy-Org's effect embeddings for H3 (the model's files: `embedding:NAME` in the prompt stands for their rows) */
const EFFECTS = ["art_is_explosion", "blooming_flowers", "bullet_time", "dark_magic", "fire_breath", "four_seasons", "kiss_camera",
                 "spiral_ascent", "storm_magic", "truman_show"];

function EffectPicker() {
  return (
    <label attrs={{ title: "an effect embedding: inserted into the prompt as embedding:NAME, where the effect should happen" }}>
      effect
      <select on={{ change: (e: Event) => {
        const el = e.target as HTMLSelectElement;
        if (el.value) form.prompt = `${form.prompt}${form.prompt.endsWith(" ") ? "" : " "}embedding:minimaxh3_${el.value} `;
        el.value = "";
        render();
      } }}>
        <option attrs={{ value: "", selected: true }}>insert…</option>
        {EFFECTS.map((n) => <option attrs={{ value: n }}>{n.replace(/_/g, " ")}</option>)}
      </select>
    </label>
  );
}

type RefKind = "refImages" | "refVideos" | "refAudios";
const REF_MAX: Record<RefKind, number> = { refImages: 9, refVideos: 3, refAudios: 3 };
const REF_ACCEPT: Record<RefKind, string> = { refImages: "image/*", refVideos: "video/*", refAudios: "audio/*" };

/** Files to the studio (POST /api/upload), each kept by the name it answers with */
async function addRefs(kind: RefKind, files: FileList | null): Promise<void> {
  for (const f of Array.from(files ?? [])) {
    if (form[kind].length >= REF_MAX[kind]) break;
    try {
      const r = await fetch(`/api/upload?name=${encodeURIComponent(f.name)}`, { method: "POST", body: f });
      const j = (await r.json()) as { ok: boolean; name?: string; error?: string };
      if (!j.ok || !j.name) throw new Error(j.error ?? `HTTP ${r.status}`);
      form[kind] = [...form[kind], j.name];
      state.error = null;
    } catch (e) {
      state.error = `${f.name}: ${String(e)}`;
    }
    render();
  }
}

/** The tags the prompt names the references by, in the order the encoder is shown them */
function refTags(): { kind: RefKind; i: number; tag: string }[] {
  const out: { kind: RefKind; i: number; tag: string }[] = [];
  form.refImages.forEach((_, i) => out.push({ kind: "refImages", i, tag: `<Picture ${i + 1}>` }));
  let audio = 0;
  form.refVideos.forEach((_, i) => {
    if (form.refVideoSound) audio += 1;
    out.push({ kind: "refVideos", i, tag: `<Video ${i + 1}>${form.refVideoSound ? ` + <Audio ${audio}>` : ""}` });
  });
  form.refAudios.forEach((_, i) => out.push({ kind: "refAudios", i, tag: `<Audio ${audio + i + 1}>` }));
  return out;
}

function References() {
  const add = (kind: RefKind, label: string) => (
    <label class="tbtn" attrs={{ title: `up to ${REF_MAX[kind]}` }}>
      <span class="i" props={{ innerHTML: "&#xf067;" }} />{label}
      <input
        attrs={{ type: "file", accept: REF_ACCEPT[kind], multiple: true, style: "display:none" }}
        on={{ change: (e: Event) => void addRefs(kind, (e.target as HTMLInputElement).files) }}
      />
    </label>
  );
  const tags = refTags();
  const files: Record<RefKind, string[]> = { refImages: form.refImages, refVideos: form.refVideos, refAudios: form.refAudios };
  return (
    <div>
      <div class="row">
        {add("refImages", "picture")}
        {add("refVideos", "clip")}
        {add("refAudios", "sound")}
        {form.refVideos.length
          ? <label class="inline"><input attrs={{ type: "checkbox", checked: form.refVideoSound }}
              on={{ change: (e: Event) => { form.refVideoSound = (e.target as HTMLInputElement).checked; render(); } }} />clips bring their sound</label>
          : null}
        {form.refImages.length
          ? <label class="inline" attrs={{ title: "2048-pixel short side instead of the clip's area: a closer likeness, several times the tokens" }}>
              <input attrs={{ type: "checkbox", checked: form.refImageMax }}
                on={{ change: (e: Event) => { form.refImageMax = (e.target as HTMLInputElement).checked; render(); } }} />full-size pictures</label>
          : null}
      </div>
      {tags.length
        ? <div class="refs">
            {tags.map((t) => (
              <span class="ref" attrs={{ title: files[t.kind][t.i] ?? "" }}>
                <a attrs={{ href: "#", title: "add the tag to the prompt" }}
                  on={{ click: (e: Event) => { e.preventDefault(); form.prompt = `${form.prompt}${form.prompt.endsWith(" ") ? "" : " "}${t.tag.split(" + ")[0]}`; render(); } }}>
                  {t.tag}
                </a>{" "}
                <span class="hint">{(files[t.kind][t.i] ?? "").replace(/^ref_\d+_/, "")}</span>
                <a attrs={{ href: "#", title: "remove" }}
                  on={{ click: (e: Event) => { e.preventDefault(); form[t.kind] = form[t.kind].filter((_, j) => j !== t.i); render(); } }}> ✕</a>
              </span>
            ))}
          </div>
        : null}
      <div class="hint">
        Name them in the prompt by their tags ("the dancer from &lt;Picture 1&gt; moves as in &lt;Video 1&gt;, speaking in the
        voice of &lt;Audio 1&gt;"). Pictures and clips switch the clip to the Ref2VA denoiser.
      </div>
    </div>
  );
}

type ControlKey = "controlVideo" | "controlMask" | "controlSource";

async function setControl(key: ControlKey, files: FileList | null): Promise<void> {
  const f = files?.[0];
  if (!f) return;
  try {
    const r = await fetch(`/api/upload?name=${encodeURIComponent(f.name)}`, { method: "POST", body: f });
    const j = (await r.json()) as { ok: boolean; name?: string; error?: string };
    if (!j.ok || !j.name) throw new Error(j.error ?? `HTTP ${r.status}`);
    form[key] = j.name;
    state.error = null;
  } catch (e) {
    state.error = `${f.name}: ${String(e)}`;
  }
  render();
}

function ControlNet() {
  const pick = (key: ControlKey, label: string, accept: string, title: string) => (
    <span class="ref" attrs={{ title }}>
      <label class="tbtn">
        <span class="i" props={{ innerHTML: "&#xf093;" }} />{label}
        <input attrs={{ type: "file", accept, style: "display:none" }}
          on={{ change: (e: Event) => void setControl(key, (e.target as HTMLInputElement).files) }} />
      </label>
      {form[key]
        ? <span> <span class="hint">{form[key].replace(/^ref_\d+_/, "")}</span>
            <a attrs={{ href: "#", title: "remove" }} on={{ click: (e: Event) => { e.preventDefault(); form[key] = ""; render(); } }}> ✕</a></span>
        : null}
    </span>
  );
  return (
    <div>
      <div class="refs">
        {pick("controlVideo", "control video", "video/*,image/*", "canny / depth / HED / MLSD / pose frames, made beforehand: the clip follows them")}
        {pick("controlMask", "mask", "video/*,image/*", "white = regenerate (a picture holds for the whole clip)")}
        {form.controlMask ? pick("controlSource", "source video", "video/*", "the video behind the mask: kept where the mask is black") : null}
      </div>
      {form.controlVideo || form.controlMask
        ? <div class="row">
            <label attrs={{ title: "how hard the control steers (1 = as trained)" }}>
              strength
              <input attrs={{ type: "number", min: 0, max: 3, step: 0.05, value: String(form.controlStrength) }}
                on={{ input: (e: Event) => { form.controlStrength = Number((e.target as HTMLInputElement).value); } }} />
            </label>
          </div>
        : null}
      <div class="hint">The Fun ControlNet-Union on the text/image-to-video denoiser (not with references). The clip's canvas and
        length are the control's: pick a canvas of its aspect.</div>
    </div>
  );
}

/** A select over the plan's names, "" first (the engine's default) */
const named = (label: string, key: "sampler" | "schedule", names: string[] | undefined, dflt: string, title: string) => (
  <label attrs={{ title }}>
    {label}
    <select on={{ change: (e: Event) => { form[key] = (e.target as HTMLSelectElement).value; render(); } }}>
      <option attrs={{ value: "", selected: form[key] === "" }}>{dflt}</option>
      {(names ?? []).map((n) => <option attrs={{ value: n, selected: form[key] === n }}>{n}</option>)}
    </select>
  </label>
);

export function Creation() {
  return (
    <div class="create">
      <textarea
        attrs={{ spellcheck: false }}
        props={{ value: form.prompt }}
        on={{ input: (e: Event) => { form.prompt = (e.target as HTMLTextAreaElement).value; } }}
      />
      <div class="hint">
        Dialogue: <code>{"(S1) says: <d>[English] line</d>"}</code> · no negations (cfg is 1.0,
        so every word renders) · 4–15 s per clip
      </div>
      <div class="row">
        {num("seconds", "seconds", 1, 15, 0.01)}
        {num("steps", "steps", 1, 40, 1)}
        {num("seed", "seed", 0, 2147483647, 1)}
        {num("upscale", "upscale", 1, 4, 0.5)}
        <label attrs={{ title: "ESRGAN: decoded at the sampled size, the frames enlarged by a network on the GPU - about half the decode stage's time (a 5 s clip at 1152x864 on a B65: 19.6 s against 41.9). latent: the latents enlarged, then decoded at the larger size - the decoder's own fine texture." }}>
          upscaler
          <select on={{ change: (e: Event) => { form.upscaler = (e.target as HTMLSelectElement).value; render(); } }}>
            <option attrs={{ value: "esrgan-general", selected: form.upscaler === "esrgan-general" }}>ESRGAN general-x4v3 (default)</option>
            <option attrs={{ value: "esrgan-anime", selected: form.upscaler === "esrgan-anime" }}>ESRGAN animevideov3 (sharper, drawn look)</option>
            <option attrs={{ value: "latent", selected: form.upscaler === "latent" }}>latent (slowest, most natural texture)</option>
          </select>
        </label>
        <EffectPicker />
        {named("sampler", "sampler", state.plan?.samplers, "euler (default)",
          "ComfyUI's samplers on the flow model. euler is the distilled default; ancestral and sde samplers add fresh noise each step; the multistep ones (dpmpp_2m, uni_pc) reuse earlier steps. Some take two or three model calls a step.")}
        {named("schedule", "schedule", state.plan?.schedules, "shifted (default)",
          "The sigma schedule. shifted is the model's own; karras and exponential suit noise-prediction models and do poorly here.")}
        <label>
          encoder
          <select on={{ change: (e: Event) => { form.te = (e.target as HTMLSelectElement).value as "teacher" | "student"; render(); } }}>
            <option attrs={{ value: "teacher", selected: form.te === "teacher" }}>teacher 32B</option>
            <option attrs={{ value: "student", selected: form.te === "student" }}>student 4B</option>
          </select>
        </label>
        <label>
          label
          <input
            attrs={{ type: "text", placeholder: "Name 01/10: …", value: form.label }}
            on={{ input: (e: Event) => { form.label = (e.target as HTMLInputElement).value; } }}
          />
        </label>
        <label class="inline">
          <input
            attrs={{ type: "checkbox", checked: form.chain }}
            on={{ change: (e: Event) => { form.chain = (e.target as HTMLInputElement).checked; render(); } }}
          />
          chain from previous clip
        </label>
      </div>
      <Panel
        id="refs"
        icon="&#xf03e;"
        title="References"
        hint={refTags().length ? refTags().map((t) => t.tag).join(" · ") : "pictures, clips, sounds the prompt names"}
      >
        <References />
      </Panel>
      <Panel
        id="control"
        icon="&#xf1de;"
        title="Control"
        hint={form.controlVideo || form.controlMask ? [form.controlVideo ? "control video" : "", form.controlMask ? "mask" : ""].filter((x) => x).join(" + ")
          : "a pose / depth / edge video to follow, or a region to regenerate"}
      >
        <ControlNet />
      </Panel>
      <Panel
        id="canvas"
        icon="&#xf0b2;"
        title="Canvas and length"
        hint={`${form.width}×${form.height} · ${form.seconds}s · ${form.steps} steps`}
      >
        <CanvasPicker
          width={form.width}
          height={form.height}
          engine={form.engine}
          seconds={form.seconds}
          steps={form.steps}
          onPick={(w, h, e, sec) => {
            form.width = w;
            form.height = h;
            form.engine = e as Engine;
            form.seconds = sec;
            render();
          }}
        />
      </Panel>
      <div class="row">
        <button type="button" on={{ click: () => void generate(false) }}>
          <span class="i" props={{ innerHTML: "&#xf0d0;" }} />Generate
        </button>
        <button type="button" class="tbtn" on={{ click: () => void generate(true) }}>
          <span class="i" props={{ innerHTML: "&#xf03a;" }} />Queue
        </button>
      </div>
      {state.error ? <pre class="err">{state.error}</pre> : null}
    </div>
  );
}
