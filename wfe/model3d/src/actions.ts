/** What the buttons do. */
import { api } from "./api.js";
import { render, saveForm, state } from "./state.js";
import type { Made } from "./types.js";

export async function refreshHistory(): Promise<void> {
  try {
    state.history = await api.history();
    if (!state.shown && state.history[0]) state.shown = state.history[0];
  } catch {
    /* transient */
  }
  state.historyLoaded = true;
  render();
}

/** A file as a data: URL */
export function readFile(f: File): Promise<string> {
  return new Promise((ok, bad) => {
    const r = new FileReader();
    r.onload = () => ok(String(r.result));
    r.onerror = () => bad(r.error);
    r.readAsDataURL(f);
  });
}

/** Pictures chosen or dropped: each a view, sides guessed in order (front, back, left, right) */
export async function addPictures(files: FileList | File[]): Promise<void> {
  const max = Math.max(state.info?.abilities?.views ?? 1, 1);
  const order = ["front", "back", "left", "right", "top", "bottom"];
  for (const f of Array.from(files)) {
    if (!f.type.startsWith("image/")) continue;
    if (max === 1) state.form.pictures = [];
    if (state.form.pictures.length >= max) break;
    const side = order.find((s) => !state.form.pictures.some((p) => p.side === s)) ?? "front";
    state.form.pictures.push({ name: f.name, data: await readFile(f), side });
  }
  render();
}

export async function chooseMesh(f: File): Promise<void> {
  state.form.mesh = await readFile(f);
  state.form.meshName = f.name;
  state.form.file = "";
  render();
}

export function show(m: Made): void {
  state.shown = m;
  render();
}

/** The request: generate from the pictures, or rig the mesh */
export async function go(): Promise<void> {
  const f = state.form;
  const body: Record<string, unknown> = { response_format: "url" };
  if (f.seed !== null) body["seed"] = f.seed;
  if (f.steps !== null) body["steps"] = f.steps;
  if (f.cfg !== null) body["cfg"] = f.cfg;
  if (Object.keys(f.options).length) body["options"] = f.options;
  saveForm();
  state.busy = true;
  state.error = null;
  state.status = state.mode === "make" ? "making the mesh…" : "rigging…";
  render();
  const t0 = performance.now();
  try {
    let a;
    if (state.mode === "make") {
      if (!f.pictures.length) throw new Error("choose a picture of the object first");
      if (f.pictures.length === 1) body["image"] = f.pictures[0]!.data;
      else body["images"] = f.pictures.map((p) => ({ image: p.data, side: p.side }));
      body["texture"] = f.texture && !!state.info?.abilities?.texture;
      if (f.resolution !== null) body["resolution"] = f.resolution;
      if (f.faces !== null) body["faces"] = f.faces;
      if (f.textureSize !== null) body["texture_size"] = f.textureSize;
      body["remove_background"] = f.removeBackground;
      a = await api.generate(body);
    } else {
      if (f.mesh) body["mesh"] = f.mesh;
      else if (f.file) body["file"] = f.file;
      else throw new Error("choose a mesh (GLB or OBJ) or one made here");
      if (f.prompt.trim()) body["prompt"] = f.prompt.trim();
      if (f.frames !== null) body["frames"] = f.frames;
      a = await api.rig(body);
    }
    const made = a.data[0];
    if (made) state.shown = made;
    state.status = `done in ${((performance.now() - t0) / 1000).toFixed(1)} s` + (made?.triangles ? ` · ${made.triangles} triangles` : "");
    await refreshHistory();
  } catch (e) {
    state.error = String(e instanceof Error ? e.message : e);
    state.status = "failed";
  } finally {
    state.busy = false;
    render();
  }
}

export async function cancel(): Promise<void> {
  await api.cancel();
  state.status = "cancelling…";
  render();
}
