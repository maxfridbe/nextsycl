/** A spinner with what is loading, for a list that has not had its first answer yet (the same in every app). `small`: inline in a header. */
import { jsx } from "../jsx.js";

export function Spinner(props: { label: string; small?: boolean }) {
  return <div class={{ spin: true, small: !!props.small }} attrs={{ role: "status", "aria-live": "polite" }}>{props.label}</div>;
}
