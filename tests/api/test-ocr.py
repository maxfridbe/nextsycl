#!/usr/bin/env python3
"""Reference test: OCR of three public sample high-school transcripts, through the front door's API alone.

Each PDF is downloaded (once, into --cache), rendered to page pictures here with pdftoppm, and only the pictures are
sent - the model never sees the PDF. What comes back is checked against what the page really says:

- **words**: the PDF's own text layer (pdftotext) is the ground truth where it has one (cairn, wtamu): the share of its
  words of 3 characters or more found in the OCR's text, counted as a multiset (a word 4 times on the page needs 4
  occurrences), order ignored (a table read row by row or column by column scores the same) and spaces ignored (a
  server that drops the table-cell tokens glues cells together) - at least --min-recall;
- **facts**: strings that must appear (names, addresses, dates, grade-point averages, course titles), after the
  layout markup is stripped and spaces folded - every one of them. hslda's page has almost no text layer (its text is
  drawn), so it is checked on facts alone, read off the page by hand;
- **no runaway**: no line repeated more than --max-repeats times (a looping decoder).

    tests/api/test-ocr.py                                  # the three transcripts, unlimited-ocr-q8, :8000
    tests/api/test-ocr.py --model unlimited-ocr-q4 --only hslda
    tests/api/test-ocr.py --api openai --url http://box:8080   # an OpenAI-style server instead (llama-server)

Needs: Python 3.9+ (no packages) and poppler's pdftoppm / pdftotext; on the server an OCR model (unlimited-ocr-*).
cairn.edu sits behind Cloudflare, which may refuse a script: put the file in --cache as cairn.pdf by hand then.
Writes into --out: each page's picture, the raw OCR text, report.json. Exits 1 when a check fails.
"""

import argparse
import base64
import collections
import html
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DOCS = {
    "cairn": {
        "url": "https://cairn.edu/documents/2019/03/home-school-transcript-example.pdf/",
        "words": True,
        "facts": ["Albert Smith", "Dr. Stan Albertson", "Sept. 1, 2014", "May 1, 2018", "Honors Precalculus",
                  "Honors Environmental Science", "Issues in American History", "Physical Education", "3.76"],
    },
    "wtamu": {
        "url": "https://www.wtamu.edu/_files/docs/admissions/Homeschool-Transcript-SAMPLE.pdf",
        "words": True,
        "facts": ["John Sample", "Home Tutorial Studies Program", "2501 4th Ave", "Canyon, TX 79016",
                  "Buster and Hannah Sample", "806-101-7896", "May 9, 2005", "Lookout Mtn", "Saxon Algebra II",
                  "French I", "Grammar & Composition", "World Literature"],
    },
    "hslda": {
        "url": "https://hslda.org/docs/librariesprovider2/public/sample-high-school-transcript-subject.pdf",
        "words": False,
        "facts": ["Mary S. Smith", "123 Main Street", "Hometown, ST", "12345", "212-343-5678", "msmith@athome.net",
                  "smithfamily@athome.net", "January 1, 2000", "June 6, 2018", "Smith Homeschool",
                  "Robert & Dora Smith", "American Lit/Comp", "British Lit/Comp", "Creative Writing", "Algebra 1",
                  "Geometry", "Physical Science with Lab", "Chemistry with Lab", "Spanish 3", "Consumer Math",
                  "Music Appreciation", "US Government", "Economics", "3.30", "ABC Community College"],
    },
}
PROMPT = "document parsing."
UA = "Mozilla/5.0 (X11; Linux x86_64; rv:140.0) Gecko/20100101 Firefox/140.0"


def token_from_disk():
    for p in ("/var/lib/nextsycl/serve.json", str(Path.home() / ".local/state/nextsycl/serve.json")):
        try:
            return json.loads(Path(p).read_text())["token"]
        except (OSError, KeyError, ValueError):
            pass
    return ""


def post(url, body, token, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise SystemExit(f"{url}: HTTP {e.code}: {e.read().decode(errors='replace')[:500]}")


def fetch(name, url, cache):
    path = cache / f"{name}.pdf"
    if path.exists() and path.read_bytes()[:5] == b"%PDF-":
        return path
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/pdf,*/*"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    if data[:5] != b"%PDF-":
        raise SystemExit(f"{name}: {url} did not send a PDF (a bot check?) - save it by hand as {path}")
    path.write_bytes(data)
    return path


def render(pdf, out, dpi):
    """The PDF's pages as PNG pictures (pdftoppm), in page order"""
    prefix = out / pdf.stem
    subprocess.run(["pdftoppm", "-r", str(dpi), "-png", str(pdf), str(prefix)], check=True)
    return sorted(out.glob(f"{pdf.stem}-*.png"))


def text_layer(pdf):
    return subprocess.run(["pdftotext", str(pdf), "-"], check=True, capture_output=True, text=True).stdout


def plain(ocr):
    """The OCR text without its layout markup: grounding tags and boxes, HTML (its tables), entities"""
    t = re.sub(r"<\|det\|>.*?<\|/det\|>", " ", ocr, flags=re.S)
    # the same with the special tokens dropped by the server: "title [76, 53, 585, 88]"
    t = re.sub(r"(?m)^\s*[a-z_]+ \[\d+, \d+, \d+, \d+\]", " ", t)
    t = re.sub(r"<\|/?(ref|grounding)\|>", " ", t)
    t = re.sub(r"<[^>]+>", " ", t)
    return html.unescape(t)


def words(t):
    """Lower-case words: letters and digits, keeping the inner . , / - of numbers and dates (3.76, 1,0, 10/2011)"""
    return [w.strip(".,/-") for w in re.findall(r"[a-z0-9]+(?:[.,/\-][a-z0-9]+)*", t.lower()) if w.strip(".,/-")]


def fold(t):
    return re.sub(r"\s+", " ", t.replace("’", "'").replace("—", "-")).strip().lower()


def ocr_page(args, token, png):
    b64 = base64.b64encode(png.read_bytes()).decode()
    t0 = time.time()
    if args.api == "openai":
        body = {"model": args.model, "temperature": 0, "max_tokens": args.max_tokens,
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": PROMPT}]}]}
        r = post(f"{args.url}/v1/chat/completions", body, token, args.timeout)
        text, tokens = r["choices"][0]["message"]["content"], r.get("usage", {}).get("completion_tokens", 0)
    else:
        body = {"model": args.model, "images": [b64], "prompt": PROMPT, "max_tokens": args.max_tokens}
        r = post(f"{args.url}/rpc/image.ocr", body, token, args.timeout)
        if not r.get("ok"):
            raise SystemExit(f"image.ocr: {r.get('error')}")
        text, tokens = r["result"]["text"], r["result"].get("tokens", 0)
    return text, tokens, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--token", default="")
    ap.add_argument("--api", choices=["nextsycl", "openai"], default="nextsycl",
                    help="nextsycl: POST /rpc/image.ocr; openai: /v1/chat/completions with an image_url part")
    ap.add_argument("--model", default="unlimited-ocr-q8")
    ap.add_argument("--only", default="", help="comma-separated: " + ", ".join(DOCS))
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--min-recall", type=float, default=0.9)
    ap.add_argument("--max-repeats", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--cache", type=Path, default=Path.home() / ".cache/nextsycl-test-ocr")
    ap.add_argument("--out", type=Path, default=Path("test-ocr-out"))
    args = ap.parse_args()
    for tool in ("pdftoppm", "pdftotext"):
        if not shutil.which(tool):
            raise SystemExit(f"{tool} is needed (poppler-utils)")
    token = args.token or token_from_disk()
    args.cache.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    names = [n.strip() for n in args.only.split(",") if n.strip()] or list(DOCS)
    report, failed = {"model": args.model, "api": args.api, "docs": {}}, False
    for name in names:
        doc = DOCS[name]
        pdf = fetch(name, doc["url"], args.cache)
        pages = render(pdf, args.out, args.dpi)
        texts, tokens, seconds = [], 0, 0.0
        for png in pages:
            t, n, s = ocr_page(args, token, png)
            texts.append(t)
            tokens += n
            seconds += s
        raw = "\n".join(texts)
        (args.out / f"{name}.ocr.txt").write_text(raw)
        got = plain(raw)
        checks = {}
        if doc["words"]:
            truth = collections.Counter(w for w in words(text_layer(pdf)) if len(w) >= 3)
            compact = re.sub(r"\s+", "", got.lower())
            seen = {w: compact.count(w) for w in truth}
            hit = sum(min(c, seen[w]) for w, c in truth.items())
            recall = hit / max(sum(truth.values()), 1)
            missing = sorted((w for w, c in truth.items() if seen[w] < c), key=lambda w: -truth[w])[:20]
            checks["words"] = {"ok": recall >= args.min_recall, "recall": round(recall, 4), "of": sum(truth.values()),
                               "missing": missing}
        g = fold(got)
        absent = [f for f in doc["facts"] if fold(f) not in g]
        checks["facts"] = {"ok": not absent, "found": len(doc["facts"]) - len(absent), "of": len(doc["facts"]),
                           "absent": absent}
        lines = collections.Counter(l.strip() for l in raw.splitlines() if l.strip())
        worst = lines.most_common(1)[0] if lines else ("", 0)
        checks["runaway"] = {"ok": worst[1] <= args.max_repeats, "most_repeated": worst[1], "line": worst[0][:80]}
        ok = all(c["ok"] for c in checks.values())
        failed |= not ok
        report["docs"][name] = {"ok": ok, "pages": len(pages), "tokens": tokens, "seconds": round(seconds, 1),
                                "tok_s": round(tokens / seconds, 1) if seconds else None, "checks": checks}
        line = f"{'PASS' if ok else 'FAIL'} {name}: {len(pages)} page(s), {tokens} tokens in {seconds:.1f} s"
        if "words" in checks:
            line += f"; words {checks['words']['recall']:.1%} of {checks['words']['of']}"
        line += f"; facts {checks['facts']['found']}/{checks['facts']['of']}"
        if absent:
            line += f" (absent: {', '.join(absent)})"
        if not checks["runaway"]["ok"]:
            line += f"; a line repeated {worst[1]} times"
        print(line, flush=True)
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
