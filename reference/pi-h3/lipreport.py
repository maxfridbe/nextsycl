"""The lip-sync test's tables and mouth sheets, from what lipsync.py and Whisper wrote beside the clips.

For each clip lip_<shot>_<model>: lipcut/<clip>.sync.json (SyncNet: offset, confidence per face track), lipcut/<clip>.wh.json
(`nextsycl audio transcribe --json` of its sound: text and word times), out/<clip>.log (the engine's "generate in N s").
Word error rate: jiwer on lower-cased, punctuation-free text against the scripted line(s) in shots.json.
Mouth sheets look/lip_<shot>.png: a row a model, for each stressed word three frames (-2, 0, +2 at 25 fps) around the word's
middle in that clip's own Whisper times (the base clip's times when Whisper missed the word), the mouth cut from SyncNet's
face box of the track that speaks it.

    python -I lipreport.py ROOT (~/src/pih3) > table.md
"""
import json
import os
import re
import sys

import jiwer
from PIL import Image, ImageDraw

ROOT = sys.argv[1]
SHOTS = json.load(open(os.path.join(ROOT, 'lip/shots.json')))
MODELS = {'fl2va': ['minimax-h3', 'minimax-h3-pi-4step', 'minimax-h3-pi-2step'],
          'ref2va': ['minimax-h3-ref2va', 'minimax-h3-pi-4step-ref2va', 'minimax-h3-pi-2step-ref2va']}
STRESS = {'A': [('keys', 0), ('table', 0), ('anytime', 0)], 'B': [('meeting', 0), ('thursday', 0), ('design', 0)],
          'C': [('tickets', 0), ('tonight', 0), ('pocket', 1), ('here', 1)], 'D': [('wind', 0), ('harbour', 0), ('dark', 0)]}


def norm(t):
    return ' '.join(re.sub(r"[^a-z0-9' ]", ' ', t.lower().replace('harbor', 'harbour').replace('any time', 'anytime')).split())


def words_of(wh):
    out = []

    def walk(v):
        if isinstance(v, dict):
            if 'start' in v and 'end' in v and ('word' in v or 'text' in v) and 'segments' not in v and 'words' not in v:
                out.append((norm(v.get('word', v.get('text', ''))), float(v['start']), float(v['end'])))
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
    walk(wh)
    return out


def load(clip):
    def p(e):
        return os.path.join(ROOT, 'lipcut', clip + e)
    sync = json.load(open(p('.sync.json'))) if os.path.exists(p('.sync.json')) and os.path.getsize(p('.sync.json')) else None
    wh = json.load(open(p('.wh.json'))) if os.path.exists(p('.wh.json')) and os.path.getsize(p('.wh.json')) else None
    secs = None
    try:
        m = re.search(r'^generate in ([0-9.]+) s', open(os.path.join(ROOT, 'out', clip + '.log')).read(), re.M)
        secs = float(m.group(1)) if m else None
    except OSError:
        pass
    return sync, wh, secs


def row(shot, model):
    clip = f'lip_{shot}_{model}'
    sync, wh, secs = load(clip)
    if sync is None:
        return None
    text = norm(wh.get('text', '')) if wh else ''
    wer = jiwer.wer(norm(SHOTS[shot]['line']), text) if wh else None
    sp = os.path.join(ROOT, 'lipcut', clip + '.span.json')
    if os.path.exists(sp) and os.path.getsize(sp):
        sync = dict(sync, tracks=json.load(open(sp))['tracks'])     # a two-shot: each speaker over its own lines
    return {'clip': clip, 'tracks': [(t['offset'], t['confidence']) for t in sync['tracks']], 'wer': wer, 'text': text, 'secs': secs,
            'sync': sync, 'words': words_of(wh) if wh else []}


def mouth(clip, box, t):
    f = max(1, int(round(t * 25)) + 1)
    path = os.path.join(ROOT, 'lipcut', clip, f'{f:05d}.png')
    if not os.path.exists(path):
        return Image.new('RGB', (120, 80), (40, 40, 40))
    im = Image.open(path).convert('RGB')
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    crop = im.crop((int(x1 - 0.05 * w), int(y1 + 0.55 * h), int(x2 + 0.05 * w), int(y2 + 0.12 * h)))
    return crop.resize((120, 80))


def sheet(shot, rows):
    base = rows[0]
    cells_w, lab = 3 * 120 + 8, 230
    W, H = lab + cells_w * len(STRESS[shot[0]]), 24 + 84 * len(rows)
    img = Image.new('RGB', (W, H), (16, 16, 16))
    d = ImageDraw.Draw(img)
    for j, (w, tr) in enumerate(STRESS[shot[0]]):
        d.text((lab + j * cells_w + 4, 4), f'"{w}"', fill=(255, 255, 0))
    for i, r in enumerate(rows):
        y = 24 + i * 84
        d.text((4, y + 4), r['clip'].split('_', 2)[2], fill=(255, 255, 255))
        d.text((4, y + 20), ' '.join(f'off {o} conf {c:.1f}' for o, c in r['tracks']), fill=(180, 220, 180))
        d.text((4, y + 36), f"WER {r['wer']:.2f}" if r['wer'] is not None else 'WER -', fill=(180, 200, 255))
        d.text((4, y + 52), f"{r['secs']:.0f} s" if r['secs'] else '', fill=(200, 200, 200))
        tracks = r['sync']['tracks']
        if not tracks:
            continue
        for j, (w, tr) in enumerate(STRESS[shot[0]]):
            hit = next((x for x in r['words'] if x[0] == w), None) or next((x for x in base['words'] if x[0] == w), None)
            tm = (hit[1] + hit[2]) / 2 if hit else 0.4 + 0.6 * j
            boxes = tracks[min(tr, len(tracks) - 1)]['boxes']
            for k, dt in enumerate((-0.08, 0.0, 0.08)):
                fi = min(len(boxes) - 1, max(0, int(round((tm + dt) * 25))))
                img.paste(mouth(r['clip'], boxes[fi], tm + dt), (lab + j * cells_w + k * 120, y))
    out = os.path.join(ROOT, 'look', f'lip_{shot}.png')
    img.save(out)
    return out


def main():
    print('| shot | model | SyncNet offset (frames) | SyncNet confidence | WER | heard | time (s) |')
    print('|---|---|---|---|---|---|---|')
    for shot in SHOTS:
        groups = ['ref2va'] if SHOTS[shot]['ref'] else ['fl2va', 'ref2va']
        rows = []
        for g in groups:
            for m in MODELS[g]:
                r = row(shot, m)
                if r is None:
                    continue
                rows.append(r)
                offs = ' / '.join(str(o) for o, c in r['tracks']) or '-'
                confs = ' / '.join(f'{c:.2f}' for o, c in r['tracks']) or '-'
                wer = f"{r['wer']:.2f}" if r['wer'] is not None else '-'
                secs = f"{r['secs']:.1f}" if r['secs'] else '-'
                print(f"| {shot} | {m} | {offs} | {confs} | {wer} | {r['text']} | {secs} |")
        if rows:
            print(f'sheet {sheet(shot, rows)}', file=sys.stderr)


main()
