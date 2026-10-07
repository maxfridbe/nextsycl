"""profile lines -> nested JSON for the flame graph"""
import json, re, sys
def load(p):
    d = {}
    for l in open(p):
        m = re.match(r"\[profile (.+?)\s{2,}([\d.]+) s\s+(\d+) calls", l)
        if m: d[m.group(1).strip()] = (float(m.group(2)), int(m.group(3)))
    return d
def node(name, s, calls=None, kids=None, note=None, est=False):
    n = {"name": name, "s": round(s, 3)}
    if calls is not None: n["calls"] = calls
    if kids: n["kids"] = kids
    if note: n["note"] = note
    if est: n["est"] = True
    return n
def kids_of(d, prefix, label=lambda k: k):
    return [node(label(k), v[0], v[1]) for k, v in sorted(d.items(), key=lambda x: -x[1][0]) if k.startswith(prefix)]
def group(d, name, prefix, cat, strip, note=None):
    s, c = d.get(name, (0, 0))
    ks = kids_of(d, prefix, lambda k: k[len(strip):])
    total = max(s, sum(k["s"] for k in ks))
    rest = total - sum(k["s"] for k in ks)
    if ks and rest > 0.05:
        ks.append(node("rest of the section", rest))
    n = node(name, total, c, ks, note); n["cat"] = cat
    for k in ks: k["cat"] = cat
    return n
def tree(d, wall, label, unit, units, decode):
    top = []
    mla = group(d, "MLA", "MLA: ", "mla", "MLA: ")
    for k in mla.get("kids", []):
        if k["name"] == "attention":
            att = kids_of(d, "MLA att: ", lambda x: x[len("MLA att: "):])
            for a in att: a["cat"] = "mla"
            if sum(a["s"] for a in att) > 0.05: k["kids"] = att
    if decode:
        s, c = d["routed experts"]
        kern = min(s, c * 0.32e-3)
        re_ = node("routed experts", s, c, [node("expert kernels (estimate)", kern, est=True, note="the grouped kernel measured alone: ~0.32 ms a 2-row layer pass"),
                                          node("waiting for copies over PCIe, second launch", s - kern, est=True, note="the rest of the section: misses and prefetches arriving")],
                   note="a verify pass's two rows through the layer's experts")
        re_["cat"] = "exp"
        for k in re_["kids"]: k["cat"] = "exp" if "kernel" in k["name"] else "wait"
    else:
        re_ = group(d, "routed experts", "MoE f16: ", "exp", "MoE f16: ", note="each expert's weights expanded to fp16, then oneMKL half GEMMs on the XMX units")
        for k in re_.get("kids", []):
            if "host copy" in k["name"]: k["cat"] = "wait"
    top.append(re_)
    top.append(group(d, "KDA", "KDA: ", "kda", "KDA: "))
    top.append(mla)
    for name, cat in [("MTP draft", "mtp"), ("hc pre", "hc"), ("shared expert", "exp"), ("GPU to GPU", "wait"), ("dense FFN", "exp"), ("router", "hc"),
                      ("expert misses (swaps / file)", "wait"), ("expert prefetch", "wait")]:
        if name in d and d[name][0] > 0.004:
            n = node(name, d[name][0], d[name][1]); n["cat"] = cat; top.append(n)
    top.sort(key=lambda n: -n["s"])
    other = wall - sum(n["s"] for n in top)
    if other > 0.05:
        o = node("other (embedding, head, host)", other, note="the pass's wall time less every section above"); o["cat"] = "other"; top.append(o)
    root = node(label, max(wall, sum(n["s"] for n in top)), None, top); root["cat"] = "root"
    root["unit"] = unit; root["units"] = units
    return root
fp, fd = load(sys.argv[1]), load(sys.argv[2])
out = {"prompt": tree(fp, 80.3, "Reading a 36,585-token prompt", "1K tokens", 36.585, False),
       "decode": tree(fd, 16.8, "Decoding 300 tokens (MTP on)", "token", 300, True)}
print(json.dumps(out))
