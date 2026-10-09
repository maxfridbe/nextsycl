/** Formatting and small numbers. */

export const fmtT = (x: number | null | undefined): string => {
  if (x === null || x === undefined) return "—";
  if (x >= 3600) return `${(x / 3600).toFixed(1)} h`;
  if (x >= 60) return `${Math.floor(x / 60)}m ${String(Math.round(x % 60)).padStart(2, "0")}s`;
  return x < 10 ? `${x.toFixed(1)}s` : `${Math.round(x)}s`;
};

/** A length as m:ss */
export const mss = (x: number): string => `${Math.floor(x / 60)}:${String(Math.round(x % 60)).padStart(2, "0")}`;

export const clamp = (x: number, lo: number, hi: number): number => Math.max(lo, Math.min(hi, x));

/** The structure tags the model knows; each goes on a line of its own. */
export const TAGS = ["intro", "verse", "pre-chorus", "chorus", "bridge", "instrumental", "solo", "outro"];

/** Length presets, seconds. */
export const LENGTHS: [string, number][] = [["0:30", 30], ["1:00", 60], ["2:00", 120], ["3:00", 180], ["4:00", 240], ["6:00", 360]];
