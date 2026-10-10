/** The API tab: the token, the spec (/openapi.yml) and every method with its curl, a click to copy. */
import { jsx } from "../jsx.js";
import { copy, ui } from "../main.js";
import type { Operation } from "../types.js";

const KIND_TITLE: Record<string, string> = { system: "System", llm: "Chat", image: "Images", audio: "Music", video: "Video", openai: "OpenAI-compatible" };

function curlOf(path: string, method: string, op: Operation, base: string, token: string): string {
  if (op["x-curl"]) return op["x-curl"];
  const ex = op.requestBody?.content["application/json"].example;
  const auth = `-H 'Authorization: Bearer ${token}'`;
  if (method === "get") return `curl -s ${base}${path.replace("{name}", "NAME")} ${auth}`;
  return `curl -s ${base}${path} ${auth} -H 'Content-Type: application/json' -d '${JSON.stringify(ex ?? {}).replace(/'/g, "'\\''")}'`;
}

function CopyBtn(text: string, key: string, label = "copy") {
  return (
    <button class="ghost small" on={{ click: () => void copy(text, key) }}>
      <span class="i" props={{ innerHTML: ui.copied === key ? "&#xf00c;" : "&#xf0c5;" }} />{ui.copied === key ? "copied" : label}
    </button>
  );
}

export function Api() {
  const sp = ui.spec;
  if (!sp) return <div class="hint">loading the spec…</div>;
  const base = sp.spec.servers[0]?.url ?? location.origin;
  const groups: Record<string, { path: string; method: string; op: Operation }[]> = {};
  for (const [path, ops] of Object.entries(sp.spec.paths)) {
    for (const [method, op] of Object.entries(ops)) {
      const tag = op.tags[0] ?? "other";
      (groups[tag] ??= []).push({ path, method, op });
    }
  }
  const order = ["system", "llm", "image", "audio", "video", "openai"];
  return (
    <div>
      <section class="panel">
        <div class="phead"><span class="i" props={{ innerHTML: "&#xf084;" }} /><span class="ptitle">Access</span></div>
        <div class="pbody">
          <div class="row">
            <span>Bearer token</span><code class="token">{sp.token}</code>{CopyBtn(sp.token, "token")}
            <a class="btn" attrs={{ href: "/openapi.yml", target: "_blank" }}><span class="i" props={{ innerHTML: "&#xf15c;" }} /> openapi.yml</a>
            {CopyBtn(`${location.origin}/openapi.yml`, "yml", "copy its URL")}
          </div>
          <div class="hint">
            Every call: <code>Authorization: Bearer &lt;token&gt;</code>. <code>POST /rpc/&lt;kind&gt;.&lt;method&gt;</code> with every variable in the
            JSON body - no path variables, no query strings - answers <code>{"{\"ok\": true, \"result\": …}"}</code> or{" "}
            <code>{"{\"ok\": false, \"error\": {\"code\", \"message\"}}"}</code>. The OpenAI-compatible <code>/v1</code> routes take OpenAI's
            bodies. Only kinds with an enabled model are listed; enabling one adds its methods.
          </div>
        </div>
      </section>
      {order.filter((t) => groups[t]).map((t) => (
        <section class="panel">
          <div class="phead"><span class="ptitle">{KIND_TITLE[t] ?? t}</span><span class="hint">{groups[t]!.length} routes</span></div>
          <div class="pbody">
            {groups[t]!.map(({ path, method, op }) => {
              const curl = curlOf(path, method, op, base, sp.token);
              const props = op.requestBody?.content["application/json"].schema?.properties ?? {};
              const req = op.requestBody?.content["application/json"].schema?.required ?? [];
              return (
                <div class="op">
                  <div class="ophead"><span class={{ verb: true, [method]: true }}>{method.toUpperCase()}</span> <code class="route">{path}</code>
                    <span class="hint"> {op.summary}</span></div>
                  {Object.keys(props).length
                    ? <div class="fields">{Object.entries(props).map(([k, v]) => (
                        <span class="field" attrs={{ title: v.description ?? "" }}><b>{k}</b>{req.includes(k) ? "*" : ""} <span class="hint">{v.type}{v.enum?.length ? `: ${v.enum.join(" | ")}` : ""}</span></span>
                      ))}</div>
                    : null}
                  <div class="curl"><pre>{curl}</pre>{CopyBtn(curl, `${method}${path}`)}</div>
                </div>
              );
            })}
          </div>
        </section>
      ))}
    </div>
  );
}
