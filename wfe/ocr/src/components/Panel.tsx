/** A collapsible region. Replaces <details>, which snabbdom cannot drive reliably: the
 *  browser owns `open` once the user clicks it, so the vdom and the DOM disagree on the
 *  next patch. Here the open flag lives in state and survives a reload. */
import { jsx } from "../jsx.js";
import { panelOpen, togglePanel } from "../state.js";
import type { VNode } from "../../../vendor/snabbdom/vnode.js";

export interface PanelProps {
  id: string;
  icon: string;
  title: string;
  /** Small grey text after the title: counts, totals, anything at a glance. */
  hint?: string;
  /** Buttons and inputs for the header. Clicks inside do not collapse the panel. */
  tools?: (VNode | string | null)[];
  /** TSX hands over a single child bare and several as an array, so accept both. */
  children?: Child | Child[];
}

type Child = VNode | string;

const kids = (c: Child | Child[] | undefined): Child[] =>
  c === undefined ? [] : Array.isArray(c) ? c : [c];

export function Panel(props: PanelProps) {
  const open = panelOpen(props.id);
  const tools = (props.tools ?? []).filter((x): x is VNode | string => x !== null);
  return (
    <section class={{ panel: true, open }}>
      <header
        class="phead"
        attrs={{ role: "button", tabindex: 0, "aria-expanded": String(open) }}
        on={{
          click: () => togglePanel(props.id),
          keydown: (e: KeyboardEvent) => {
            if (e.key === "Enter" || e.key === " ") {
              e.preventDefault();
              togglePanel(props.id);
            }
          },
        }}
      >
        <span class="chev">{open ? "▾" : "▸"}</span>
        <span class="i" props={{ innerHTML: props.icon }} />
        <span class="ptitle">{props.title}</span>
        {props.hint ? <span class="hint">{props.hint}</span> : null}
        {tools.length
          ? <span class="ptools" on={{ click: (e: Event) => e.stopPropagation() }}>{tools}</span>
          : null}
      </header>
      {open ? <div class="pbody">{kids(props.children)}</div> : null}
    </section>
  );
}
