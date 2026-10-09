/** Formatting and small numbers. */

export const fmtT = (x: number | null | undefined): string => {
  if (x === null || x === undefined) return "—";
  if (x >= 3600) return `${(x / 3600).toFixed(1)} h`;
  if (x >= 60) return `${Math.floor(x / 60)}m ${String(Math.round(x % 60)).padStart(2, "0")}s`;
  return x < 10 ? `${x.toFixed(1)}s` : `${Math.round(x)}s`;
};

/** A side to the model's grid: a multiple of 16, at least 256. */
export const r16 = (x: number): number => Math.max(256, Math.round(x / 16) * 16);

export const clamp = (x: number, lo: number, hi: number): number => Math.max(lo, Math.min(hi, x));

/** The aspect presets: name, width part, height part. */
export const ASPECTS: [string, number, number][] = [
  ["1:1", 1, 1], ["4:3", 4, 3], ["3:4", 3, 4], ["3:2", 3, 2], ["2:3", 2, 3], ["16:9", 16, 9], ["9:16", 9, 16], ["21:9", 21, 9],
];
