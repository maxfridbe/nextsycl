/** Songs: the last request's, open; the ones made here, a list - each with its player, settings and lyrics. */
import { jsx } from "../jsx.js";
import { reuse, saveVoice } from "../actions.js";
import { render, state } from "../state.js";
import type { Song } from "../types.js";
import { fmtT, mss } from "../util.js";

function meta(s: Song): string {
  const who = s.cloned ? "cloned voice · " : s.voice ? `${s.voice} · ` : "";
  const lang = s.language ? `${s.language} · ` : "";
  const rt = s.took ? ` · made in ${fmtT(s.took)} (${(s.seconds / s.took).toFixed(2)}×)` : "";
  const music = s.cloned === undefined;
  return `${who}${lang}${mss(s.seconds)} · seed ${s.seed}${music && s.steps ? ` · ${s.steps} steps` : ""}${music && s.cfg ? ` · cfg ${+s.cfg.toFixed(2)}` : ""}${rt}` +
    (s.wh != null ? ` · ${s.wh.toFixed(2)} Wh` : "");
}

export function SongCard(props: { s: Song; big: boolean }) {
  const s = props.s;
  const shown = props.big || !!state.shown[s.url];
  return (
    <div class={{ song: true, big: props.big }}>
      <div class="desc" attrs={{ title: s.prompt }}>{s.prompt}</div>
      {s.instructions ? <div class="hint" attrs={{ title: s.instructions }}>{s.instructions}</div> : null}
      <audio attrs={{ src: s.url, controls: true, preload: props.big ? "auto" : "none" }} />
      <div class="meta">
        <span>{meta(s)}</span>
        <span class="acts">
          {s.lyrics && !props.big
            ? <button type="button" class="tbtn" on={{ click: () => { state.shown[s.url] = !shown; render(); } }}>{shown ? "hide lyrics" : "lyrics"}</button>
            : null}
          <button type="button" class="tbtn" attrs={{ title: "use this description, lyrics and seed again" }} on={{ click: () => reuse(s) }}>reuse</button>
          {s.cloned !== undefined && state.info?.speech
            ? <button type="button" class="tbtn" attrs={{ title: "keep this voice by name: any model that clones speaks it" }}
                      on={{ click: () => void saveVoice({ file: s.url, description: s.instructions }) }}>save voice…</button>
            : null}
          <a class="tbtn" attrs={{ href: s.url, download: s.url.split("/").pop() ?? "song.wav" }}>save</a>
        </span>
      </div>
      {s.lyrics && shown ? <pre class="lyr">{s.lyrics}</pre> : null}
    </div>
  );
}

export function Songs() {
  // a speech server lists its speech, a song server its songs (they share the output directory)
  const speech = !!state.info?.speech;
  const list = state.history.filter((s) => (s.cloned !== undefined) === speech);
  if (!list.length) return <div class="hint">Nothing made yet. Ctrl+Enter in a text box generates.</div>;
  return <div class="songs">{list.map((s) => <SongCard s={s} big={false} />)}</div>;
}
