/** Hash routing, so the back button works and a clip has a link you can paste.
 *
 *    #!/create                    the creation tab
 *    #!/jobs                      the jobs tab, project chosen automatically
 *    #!/jobs/Patter%20dance       that project's timeline
 *    #!/jobs/Patter%20dance/12    with clip 12 selected, scrolled into view
 *
 *  One direction only: a click calls navigate(), which sets location.hash; the hashchange
 *  event calls applyRoute(), which is the only thing that mutates the tab or the selection.
 *  That keeps the URL and the screen from disagreeing, which is what a second code path
 *  would eventually cause.
 */
import { loadTimeline, selectClip } from "./project.js";
import { render, rememberTab, state, type Tab } from "./state.js";

export interface Route {
  tab: Tab;
  project: string | null;
  clip: number | null;
}

export function parseHash(hash: string): Route {
  const parts = hash.replace(/^#!?\/?/, "").split("/").filter(Boolean);
  let decoded: string[];
  try {
    decoded = parts.map(decodeURIComponent);
  } catch {
    decoded = parts; // a hand-typed hash with a stray % should not blank the page
  }
  if (decoded[0] === "create") return { tab: "create", project: null, clip: null };
  if (decoded[0] !== "jobs") return { tab: state.tab, project: null, clip: null };
  const n = decoded[2] === undefined ? NaN : parseInt(decoded[2], 10);
  return {
    tab: "jobs",
    project: decoded[1] ?? null,
    clip: Number.isFinite(n) ? n : null,
  };
}

export function hashFor(r: Route): string {
  if (r.tab === "create") return "#!/create";
  const parts = ["#!", "jobs"];
  if (r.project) {
    parts.push(encodeURIComponent(r.project));
    if (r.clip !== null) parts.push(String(r.clip));
  }
  return parts.join("/");
}

export function currentRoute(): Route {
  return { tab: state.tab, project: state.timelineProject, clip: state.selected };
}

/** Change part of the route. Writing the hash is what triggers the work, via hashchange. */
export function navigate(patch: Partial<Route>, replace = false): void {
  const next = { ...currentRoute(), ...patch };
  const hash = hashFor(next);
  if (location.hash === hash) {
    void applyRoute(next); // same URL, but state may have drifted (project reloaded, say)
    return;
  }
  if (replace) location.replace(`${location.pathname}${location.search}${hash}`);
  else location.hash = hash;
}

let applying = false;

export async function applyRoute(r: Route): Promise<void> {
  if (applying) return;
  applying = true;
  try {
    state.tab = r.tab;
    rememberTab(r.tab);
    if (r.tab === "jobs") {
      if (r.project && r.project !== state.timelineProject) await loadTimeline(r.project);
      if (r.clip !== state.selected) await selectClip(r.clip);
    }
    render();
  } finally {
    applying = false;
  }
}

/** Put the current state in the URL without adding a history entry. Used when the page
 *  picks a project by itself, so the back button does not walk through choices the user
 *  never made. */
export function syncHash(): void {
  const hash = hashFor(currentRoute());
  if (location.hash !== hash) {
    location.replace(`${location.pathname}${location.search}${hash}`);
  }
}

export function startRouter(): void {
  window.addEventListener("hashchange", () => void applyRoute(parseHash(location.hash)));
  if (location.hash) void applyRoute(parseHash(location.hash));
  else syncHash(); // first visit: adopt the remembered tab rather than leaving a bare URL
}
