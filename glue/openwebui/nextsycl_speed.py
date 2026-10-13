"""
title: nextsycl speed
author: nextsycl
description: Shows prompt reading (pp) and generation (tg) speeds above each answer, live, as the nextsycl chat server reports them (the front door's llm.requests).
version: 0.1.0
"""

# An Open WebUI filter function (Open WebUI's plugins are Python; nextsycl itself has none). Install it in Admin
# Panel > Functions (or glue/openwebui/install.sh), enable it globally, and set its valves: the front door's url and
# its API token (the :8000 page's API tab).

import asyncio
import time

import aiohttp
from pydantic import BaseModel, Field


class Filter:
    class Valves(BaseModel):
        priority: int = Field(default=0, description="the filter's order")
        url: str = Field(default="http://127.0.0.1:8000", description="the nextsycl front door")
        token: str = Field(default="", description="its API token (the :8000 page's API tab)")
        interval: float = Field(default=0.5, description="seconds between updates")

    def __init__(self):
        self.valves = self.Valves()
        # the server's request ids already shown under an answer
        self.claimed: set = set()

    async def inlet(self, body: dict, __event_emitter__=None, __metadata__=None) -> dict:
        meta = __metadata__ or {}
        # Open WebUI's own background calls (titles, tags, follow-ups) have no answer to show it on
        if __event_emitter__ is not None and not meta.get("task"):
            asyncio.create_task(self.watch(str(body.get("model", "")), time.time(), __event_emitter__))
        return body

    async def requests(self, s: aiohttp.ClientSession) -> dict:
        async with s.post(f"{self.valves.url.rstrip('/')}/rpc/llm.requests", json={}) as r:
            v = await r.json(content_type=None)
            if not v.get("ok"):
                raise RuntimeError(v.get("error", {}).get("message", "llm.requests failed"))
            return v["result"]

    def mine(self, rows: list, model: str, t0: float):
        def same(m: str) -> bool:
            return not model or not m or m == model or m.endswith(model) or model.endswith(m)

        new = [r for r in rows if r.get("started", 0) >= int(t0) - 1 and r.get("id") not in self.claimed and same(str(r.get("model", "")))]
        return min(new, key=lambda r: (r.get("started", 0), r.get("id", 0))) if new else None

    async def watch(self, model: str, t0: float, emit) -> None:
        headers = {"Authorization": f"Bearer {self.valves.token}"}
        rid, last, fails = None, "", 0
        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(headers=headers, timeout=timeout) as s:
            # a model loading (a swap, R1 from disk) can take minutes before its request shows
            while time.time() - t0 < 3600:
                await asyncio.sleep(self.valves.interval)
                try:
                    v = await self.requests(s)
                    fails = 0
                except Exception:
                    fails += 1
                    if fails > 20:
                        return
                    continue
                active, done = v.get("active", []), v.get("done", [])
                if rid is None:
                    r = self.mine(active, model, t0) or self.mine(done, model, t0)
                    if r is None:
                        if time.time() - t0 > 2 and last != "waiting":
                            last = "waiting"
                            await emit({"type": "status", "data": {"description": "waiting for the model", "done": False}})
                        continue
                    rid = r["id"]
                    self.claimed.add(rid)
                    if len(self.claimed) > 1000:
                        self.claimed = set(sorted(self.claimed)[-500:])
                row = next((r for r in active if r.get("id") == rid), None)
                ended = row is None
                if ended:
                    row = next((r for r in done if r.get("id") == rid), None)
                    if row is None:
                        return
                text = describe(row)
                if text != last or ended:
                    last = text
                    await emit({"type": "status", "data": {"description": text, "done": ended}})
                if ended:
                    return


def describe(r: dict) -> str:
    prompt, reused = int(r.get("prompt_tokens") or 0), int(r.get("reused") or 0)
    state = r.get("state", "")
    if state == "waiting":
        return f"queued: {prompt:,}-token prompt"
    cached = f", {reused:,} cached" if reused else ""
    if state == "reading":
        read = int(r.get("read") or reused)
        return f"pp {float(r.get('read_tok_s') or 0):,.0f} tok/s  ({read - reused:,} of {prompt - reused:,} tokens{cached})"
    secs = float(r.get("read_seconds") or 0)
    fed = prompt - reused
    pp = f"pp {fed / secs:,.0f} tok/s ({fed:,} tok in {secs:.1f} s{cached})" if secs > 0 and fed > 1 else f"pp cached ({prompt:,} tok)"
    n = int(r.get("generated") or 0)
    tg = f"tg {float(r.get('tok_s') or 0):.1f} tok/s ({n:,} tok)"
    drafts = r.get("drafts")
    if isinstance(drafts, list) and len(drafts) == 2 and drafts[1]:
        tg += f", drafts {drafts[0]}/{drafts[1]}"
    return f"{pp}  ·  {tg}"
