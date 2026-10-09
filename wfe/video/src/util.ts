/** Formatting helpers. Anything that turns a number into words lives here. */

export const fmtMS = (x: number | null | undefined): string =>
  x === null || x === undefined
    ? "—"
    : `${Math.floor(x / 60)}m ${String(Math.round(x % 60)).padStart(2, "0")}s`;

export const fmtT = (x: number | null | undefined): string => {
  if (x === null || x === undefined) return "—";
  if (x >= 3600) return `${(x / 3600).toFixed(1)} h`;
  if (x >= 60) return `${Math.floor(x / 60)}m ${String(Math.round(x % 60)).padStart(2, "0")}s`;
  return `${Math.round(x)}s`;
};

export const clipNumber = (label: string): number => {
  const m = /\s(\d+)\//.exec(label);
  return m?.[1] ? parseInt(m[1], 10) : 0;
};

/** Label prefix before the "NN/MM" counter - what a project is called when nothing says. */
export const labelPrefix = (label: string): string =>
  (/^(.*?)\s*\d+\//.exec(label)?.[1] ?? "").trim();

export const clamp = (x: number, lo: number, hi: number): number => Math.max(lo, Math.min(hi, x));
