import json, sys
r = json.load(open(sys.argv[1]))
c = r["choices"][0]
print("thinking:", repr(c["message"]["reasoning_content"][:300]))
print("answer:  ", repr(c["message"]["content"]))
print("usage:   ", r["usage"])
for e in c["logprobs"]["content"]:
    ch = e.get("chain", {})
    print(f"\n{e['token']!r:10} logprob {e['logprob']:8.4f}   chained {ch.get('logprob', float('nan')):8.4f}   attention to this turn {ch.get('attention_to_turn', 0):.3f}")
    for x in ch.get("echo", []):
        print(f"     echoes turn token #{x['turn_position']} {x['token']!r}: attention {x['attention']:.4f} x its logprob {x['logprob']:.4f}")
    print("     alternatives: " + ", ".join(f"{t['token']!r} {t['logprob']:.2f} -> {t.get('chained_logprob', float('nan')):.2f}" for t in e["top_logprobs"]))
