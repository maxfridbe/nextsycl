/** Formatting and small numbers. */

export const fmtT = (x: number | null | undefined): string => {
  if (x === null || x === undefined) return "—";
  if (x >= 3600) return `${(x / 3600).toFixed(1)} h`;
  if (x >= 60) return `${Math.floor(x / 60)}m ${String(Math.round(x % 60)).padStart(2, "0")}s`;
  return x < 10 ? `${x.toFixed(1)}s` : `${Math.round(x)}s`;
};

/** A side to the model's grid (the engine's: 16, HunyuanImage's 32): a multiple of it, at least 256. */
export const snap = (x: number, grid: number = 16, min: number = 256): number => Math.max(min, Math.round(x / grid) * grid);

/** A size from a pixel budget at an aspect ratio (W / H): the budget's root, each side rounded to the grid (halves
 *  up, as OpenMOSS's fit_size), at least one grid step and at most `max`. */
export const fitSize = (pixels: number, ratio: number, grid: number = 16, max: number = 4096): [number, number] => {
  const side = (e: number): number => Math.min(max, Math.max(grid, Math.floor(e / grid + 0.5) * grid));
  return [side(Math.sqrt(pixels * ratio)), side(Math.sqrt(pixels / ratio))];
};

/** A ratio "W:H" as a number. */
export const ratioOf = (r: string): number => {
  const [w = 1, h = 1] = r.split(":").map(Number);
  return w / h;
};

/** A budget's label: below 1 M the square's side, else millions of pixels. */
export const pixelLabel = (p: number): string => (p < 1048576 ? `${Math.round(Math.sqrt(p))}²` : `${+(p / 1048576).toFixed(2)}M`);

export const clamp = (x: number, lo: number, hi: number): number => Math.max(lo, Math.min(hi, x));

/** The aspect presets: name, width part, height part. */
export const ASPECTS: [string, number, number][] = [
  ["1:1", 1, 1], ["4:3", 4, 3], ["3:4", 3, 4], ["3:2", 3, 2], ["2:3", 2, 3], ["16:9", 16, 9], ["9:16", 9, 16], ["21:9", 21, 9],
];
