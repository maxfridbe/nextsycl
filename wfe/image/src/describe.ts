/** The Describe panel's rules, apart from the page so a test can run them: which prompt styles there are and which read
 *  a negative prompt, the family of the model the page serves, the size a description suggests when the form can take it,
 *  and an error in words. (docs/api/image-describe.md is the contract; the server's table is glue/serve/src/describe.rs.) */
import type { DescribeJob, Info } from "./types.js";

/** The prompt styles (`target`): name, whether its models read a negative prompt, what it is for. */
export const FAMILIES: { id: string; negative: boolean; about: string }[] = [
  { id: "natural", negative: true, about: "plain sentences, for any model" },
  { id: "qwen-image", negative: true, about: "Qwen-Image 2.1" },
  { id: "z-image", negative: false, about: "Z-Image-Turbo" },
  { id: "h3", negative: false, about: "P-H3-Image, MiniMax-H3-Image" },
  { id: "hunyuanimage", negative: true, about: "HunyuanImage 2.1" },
  { id: "moss-image", negative: true, about: "MOSS-Image" },
  { id: "ming-image", negative: false, about: "Ming-Image design" },
  { id: "tags", negative: true, about: "comma-separated tags" },
];

/** The family of a registry architecture (the server's Family::of_arch); "natural" for one it does not know. */
export function familyOfArch(arch: string | undefined): string {
  switch (arch) {
    case "qwen-image-2.1": return "qwen-image";
    case "z-image": return "z-image";
    case "p-h3-image": case "minimax-h3-image": return "h3";
    case "hunyuanimage-2.1": return "hunyuanimage";
    case "moss-image": return "moss-image";
    case "ming-image": return "ming-image";
    default: return "natural";
  }
}

/** What the request names as `target`: the form's choice, else the model this page serves (its id is a target). */
export function targetOf(choice: string, info: Info | null): string {
  return choice || info?.model || "natural";
}

/** The family a target is for: a family's own name, else the page's model's architecture. */
export function familyOf(target: string, info: Info | null): string {
  if (FAMILIES.some((f) => f.id === target)) return target;
  return target === info?.model ? familyOfArch(info?.arch) : "natural";
}

/** Whether the target's models read a negative prompt: the family's rule, and for the page's own model what it reports
 *  (`guidance`: a distilled model takes none). */
export function takesNegative(target: string, info: Info | null): boolean {
  if (target === info?.model && info?.guidance !== undefined) return info.guidance;
  return FAMILIES.find((f) => f.id === familyOf(target, info))?.negative ?? false;
}

/** The size a description suggests, when it is for this page's model (or its family) and the sides are on the page's
 *  grid and in its range; else none. */
export function sizeToApply(sug: { width: number; height: number } | null | undefined, target: string, info: Info | null): [number, number] | null {
  if (!sug || !info) return null;
  if (target !== info.model && familyOf(target, info) !== familyOfArch(info.arch)) return null;
  const g = info.grid ?? 16;
  const lo = info.sizes ? 64 : 256;
  const ok = (x: number): boolean => x % g === 0 && x >= lo && x <= 4096;
  return ok(sug.width) && ok(sug.height) ? [sug.width, sug.height] : null;
}

/** An error in words: its code is the first thing the reader needs ("conflict" is a model not enabled). */
export function errorText(e: { code?: string; message?: string; field?: string | null } | undefined): string {
  if (!e) return "the describe failed";
  const m = e.message ?? "the describe failed";
  switch (e.code) {
    case "conflict": return `${m}\n(409: no model that can describe is enabled for this choice - enable one, or pick another describer or family)`;
    case "model_not_downloaded": return `${m}\n(the describer is a card's, not registered here yet)`;
    case "no_room": return `${m}\n(no room on a card for the describer now: try again when the box is less busy)`;
    case "cancelled": return "cancelled";
    default: return e.code ? `${e.code}: ${m}${e.field ? ` (field ${e.field})` : ""}` : m;
  }
}

/** Where a job is, in words, for the progress line. */
export function jobStage(j: DescribeJob | null, seconds: number): string {
  if (!j) return "sending the picture";
  if (j.state === "queued") return j.waiting_for ? `waiting: ${j.waiting_for}` : j.position ? `queued, place ${j.position}` : "queued";
  const cold = seconds > 8 ? " - the first describe loads the describer (about 45 s cold)" : "";
  return `${j.stage === "loading" ? "loading the describer" : "reading the picture"}${cold}`;
}
