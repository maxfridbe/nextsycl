"""A contact sheet for one shot: its reference pictures in the first row, then one labelled row a clip (rows made
beforehand, e.g. 4 frames side by side), all at the same width.

    python -I stack.py out.png --refs a.png,b.png "Ref2VA 15 steps=row1.png" "pi 2 steps=row2.png" ...
"""
import argparse

from PIL import Image, ImageDraw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--refs', required=True)
    ap.add_argument('rows', nargs='+')
    a = ap.parse_args()
    rows = [(r.split('=', 1)[0], Image.open(r.split('=', 1)[1]).convert('RGB')) for r in a.rows]
    w = max(im.width for _, im in rows)
    refs = [Image.open(p).convert('RGB') for p in a.refs.split(',')]
    h = 240
    refs = [im.resize((max(1, im.width * h // im.height), h)) for im in refs]
    lab = 28
    total = lab + h + sum(lab + im.height * w // im.width for _, im in rows)
    sheet = Image.new('RGB', (w, total), 'white')
    d = ImageDraw.Draw(sheet)
    y = 0
    d.text((8, y + 6), 'references', fill='black')
    x = 0
    for im in refs:
        sheet.paste(im, (x, y + lab))
        x += im.width + 8
    y += lab + h
    for name, im in rows:
        d.text((8, y + 6), name, fill='black')
        im = im.resize((w, im.height * w // im.width))
        sheet.paste(im, (0, y + lab))
        y += lab + im.height
    sheet.save(a.out)


if __name__ == '__main__':
    main()
