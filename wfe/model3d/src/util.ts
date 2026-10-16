/** Formatting. */

export const fmtT = (x: number | null | undefined): string => {
  if (x === null || x === undefined) return "—";
  if (x >= 3600) return `${(x / 3600).toFixed(1)} h`;
  if (x >= 60) return `${Math.floor(x / 60)}m ${String(Math.round(x % 60)).padStart(2, "0")}s`;
  return x < 10 ? `${x.toFixed(1)}s` : `${Math.round(x)}s`;
};

/** 123456 -> "123k" */
export const kilo = (n: number | null | undefined): string =>
  n === null || n === undefined ? "—" : n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `${Math.round(n / 1e3)}k` : String(n);

export const bytes = (n: number | undefined): string =>
  n === undefined ? "" : n >= 1 << 20 ? `${(n / (1 << 20)).toFixed(1)} MB` : `${Math.round(n / 1024)} KB`;

/** The sides a view can show */
export const SIDES = ["front", "back", "left", "right", "top", "bottom"];
