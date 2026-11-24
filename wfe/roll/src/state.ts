/** The page's state and its redraw hook (H3's scheme: one state object, render() schedules a patch). */

export interface Ev {
  prompt: string;
  seconds: number;
}

export const MODELS = ["movieroll-distilled-t2v-q8", "movieroll-distilled-i2v-q8", "movieroll-base-t2v-q8"];

export const state = {
  model: MODELS[0] as string,
  label: `roll-${new Date().toISOString().slice(0, 19).replace(/[-:T]/g, "")}`,
  seed: 7,
  live: true,
  maxIdle: 300,
  firstFrame: "",
  events: [
    { prompt: "A red fox with a thick orange coat trots steadily through deep snow in a quiet pine forest at dawn. The camera tracks the red fox, continuously keeping the red fox centered. Cinematic photography realistic style.", seconds: 5 },
  ] as Ev[],
  add: { prompt: "", seconds: 5 } as Ev,
  /** the film this page started or follows: its label */
  film: null as string | null,
  status: null as Record<string, any> | null,
  busy: false,
  error: null as string | null,
  said: [] as string[],
};

let hook: () => void = () => {};
export function onRender(f: () => void): void {
  hook = f;
}
export function render(): void {
  hook();
}

/** A line in the page's own log (what it asked and what came back) */
export function say(line: string): void {
  state.said.push(`${new Date().toLocaleTimeString()}  ${line}`);
  if (state.said.length > 60) state.said.shift();
  render();
}
