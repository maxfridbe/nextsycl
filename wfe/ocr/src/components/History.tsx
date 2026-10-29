/** The readings made since the server started, newest first: click one to show it. */
import { jsx } from "../jsx.js";
import { show } from "../actions.js";
import { plain } from "../ocr.js";
import { state } from "../state.js";
import { fmtT } from "../util.js";
import { Spinner } from "./Spinner.js";

export function History() {
  if (!state.historyLoaded) return <Spinner label="Loading the readings…" />;
  if (!state.history.length) return <div class="hint">Nothing read yet.</div>;
  return (
    <div class="hist">
      {state.history.map((r) => (
        <div class="hitem" on={{ click: () => show(r) }}>
          <span class="t">{plain(r.text).replace(/\s+/g, " ").slice(0, 160)}</span>
          <span class="m">{`${r.pages} p · ${r.tokens} tok · ${fmtT(r.seconds)}${r.created ? ` · ${new Date(r.created * 1000).toLocaleTimeString()}` : ""}`}</span>
        </div>
      ))}
    </div>
  );
}
