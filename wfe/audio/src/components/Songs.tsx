/** Songs: the last request's, open; the ones made here, a list - each with its player, settings and lyrics. */
import { jsx } from "../jsx.js";
import { reuse } from "../actions.js";
import { render, state } from "../state.js";
import type { Song } from "../types.js";
import { fmtT, mss } from "../util.js";

function meta(s: Song): string {
  const rt = s.took ? ` · made in ${fmtT(s.took)} (${(s.seconds / s.took).toFixed(2)}×)` : "";
  return `${mss(s.seconds)} · seed ${s.seed}${s.steps ? ` · ${s.steps} steps` : ""}${s.cfg ? ` · cfg ${+s.cfg.toFixed(2)}` : ""}${rt}` +
    (s.wh != null ? ` · ${s.wh.toFixed(2)} Wh` : "");
}

export function SongCard(props: { s: Song; big: boolean }) {
  const s = props.s;
  const shown = props.big || !!state.shown[s.url];
  return (
    <div class={{ song: true, big: props.big }}>
      <div class="desc" attrs={{ title: s.prompt }}>{s.prompt}</div>
      <audio attrs={{ src: s.url, controls: true, preload: props.big ? "auto" : "none" }} />
      <div class="meta">
        <span>{meta(s)}</span>
        <span class="acts">
          {s.lyrics && !props.big
            ? <button type="button" class="tbtn" on={{ click: () => { state.shown[s.url] = !shown; render(); } }}>{shown ? "hide lyrics" : "lyrics"}</button>
            : null}
          <button type="button" class="tbtn" attrs={{ title: "use this description, lyrics and seed again" }} on={{ click: () => reuse(s) }}>reuse</button>
          <a class="tbtn" attrs={{ href: s.url, download: s.url.split("/").pop() ?? "song.wav" }}>save</a>
        </span>
      </div>
      {s.lyrics && shown ? <pre class="lyr">{s.lyrics}</pre> : null}
    </div>
  );
}

export function Songs() {
  if (!state.history.length) return <div class="hint">Nothing made yet. Ctrl+Enter in a text box generates.</div>;
  return <div class="songs">{state.history.map((s) => <SongCard s={s} big={false} />)}</div>;
}
