/** JSX factory for snabbdom. tsconfig points jsxFactory here, so `<div/>` becomes jsx(...).
 *
 *  Two conveniences over raw snabbdom, both resolved at build time:
 *    class="a b"            -> folded into the selector (h("div.a.b"))
 *    class={{ a: cond }}    -> snabbdom's class module, unchanged
 */
import { h } from "../../vendor/snabbdom/h.js";
import type { VNode, VNodeData } from "../../vendor/snabbdom/vnode.js";
import type { Classes } from "../../vendor/snabbdom/modules/class.js";

type Child = VNode | string | number | null | undefined | boolean | Child[];

export type JsxData = Omit<VNodeData, "class"> & {
  class?: string | Classes;
} & Record<string, unknown>;

function flatten(children: Child[], out: (VNode | string)[] = []): (VNode | string)[] {
  for (const c of children) {
    if (c === null || c === undefined || c === false || c === true) continue;
    if (Array.isArray(c)) flatten(c, out);
    else out.push(typeof c === "object" ? c : String(c));
  }
  return out;
}

export function jsx(
  tag: string | ((props: Record<string, unknown>) => VNode),
  data: JsxData | null,
  ...children: Child[]
): VNode {
  const kids = flatten(children);
  if (typeof tag === "function") return tag({ ...(data ?? {}), children: kids });

  const d: JsxData = { ...(data ?? {}) };
  let sel = tag;
  if (typeof d.class === "string") {
    const names = d.class.split(/\s+/).filter(Boolean);
    if (names.length) sel = `${tag}.${names.join(".")}`;
    delete d.class;
  }
  return h(sel, d as VNodeData, kids);
}

export function Fragment(props: { children?: (VNode | string)[] }): VNode {
  return h("div", {}, props.children ?? []);
}

declare global {
  namespace JSX {
    type Element = VNode;
    interface ElementChildrenAttribute {
      children: object;
    }
    interface IntrinsicElements {
      [tag: string]: JsxData;
    }
  }
}
