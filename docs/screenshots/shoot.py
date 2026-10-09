"""docs/screenshots/shoot.py: the README's screenshots of a kind's web front end, from a running server - for a quick
look at what a page should show when debugging it. Needs playwright (pip install playwright; python -m playwright
install chromium); the page renders through requestAnimationFrame, so a plain headless browser screenshot comes out
blank - this waits for the panels.

    python shoot.py image http://localhost:8086 docs/screenshots   # idle, running, done; desktop and phone widths;
                                                                   # NS_SHOT_PICTURES=a.png,b.png: an edit too
    python shoot.py audio http://localhost:8087 docs/screenshots   # idle, composing, done; desktop and phone widths
    python shoot.py video http://localhost:8095 docs/screenshots   # the studio (nextsycl video serve)

Written as PNG, kept in the repository as WebP (quality 82: a tenth of the size).
"""
import asyncio
import os
import sys

from playwright.async_api import async_playwright

PROMPT = "a lighthouse on a rocky point at dusk, oil painting, heavy brush strokes"


async def image(p, url, out):
    b = await p.chromium.launch()
    pg = await b.new_page(viewport={"width": 1400, "height": 1000})
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    await pg.goto(url)
    await pg.wait_for_selector("section.panel", timeout=60000)
    await pg.fill("textarea", PROMPT)
    await pg.wait_for_timeout(800)
    await pg.screenshot(path=f"{out}/image-wfe-idle.png")
    await pg.click("#go")
    await pg.wait_for_function("document.querySelector('.clipprog .bar .fill') && "
                               "parseFloat(document.querySelector('.clipprog .bar .fill').style.width) > 30", timeout=180000)
    await pg.screenshot(path=f"{out}/image-wfe-running.png")
    await pg.wait_for_selector("figure.pic.big", timeout=300000)
    await pg.wait_for_timeout(1500)
    await pg.screenshot(path=f"{out}/image-wfe-done.png", full_page=True)
    await pg.set_viewport_size({"width": 390, "height": 844})
    await pg.wait_for_timeout(800)
    await pg.screenshot(path=f"{out}/image-wfe-phone.png", full_page=False)
    # an edit: two pictures (NS_SHOT_PICTURES: two files, comma separated), instructions naming them
    pics = os.environ.get("NS_SHOT_PICTURES")
    if pics:
        await pg.set_viewport_size({"width": 1400, "height": 1000})
        if not await pg.is_visible(".drop"):
            await pg.click("text=Pictures")
        await pg.set_input_files(".drop input[type=file]", pics.split(","))
        await pg.wait_for_function("document.querySelectorAll('.drop figure.pic').length >= 2", timeout=30000)
        await pg.fill("textarea", "Place the fox from picture 1 on the rocks in front of the lighthouse from picture 2, at dusk")
        await pg.click("#go")
        await pg.wait_for_function("document.querySelector('#go') && document.querySelector('#go').textContent.includes('Edit')", timeout=600000)
        await pg.wait_for_selector("figure.pic.big", timeout=600000)
        await pg.wait_for_timeout(1500)
        await pg.screenshot(path=f"{out}/image-wfe-edit.png", full_page=True)
    await b.close()
    return errors


DESCRIPTION = ("Genre: indie folk. BPM: 84. Key: G major. Warm, nostalgic, building to a full-band chorus. Vocals: male lead, "
               "gentle and slightly husky, harmonies in the chorus. Arrangement: acoustic guitar and banjo, upright bass, "
               "soft drums entering in the chorus.")
LYRICS = ("[verse]\nThe kettle sings at half past six\nThe dog is dreaming by the door\n[chorus]\nAnd every road I ever took\n"
          "Was leading me back home once more")


async def audio(p, url, out):
    b = await p.chromium.launch()
    pg = await b.new_page(viewport={"width": 1300, "height": 1000})
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    await pg.goto(url)
    await pg.wait_for_selector("section.panel", timeout=60000)
    await pg.fill(".create textarea:not(.lyrics)", DESCRIPTION)
    await pg.fill("textarea.lyrics", LYRICS)
    await pg.click(".create .chip:text-is(\"0:30\")")
    await pg.wait_for_timeout(800)
    await pg.screenshot(path=f"{out}/audio-wfe-idle.png")
    await pg.click("#go")
    await pg.wait_for_function("document.querySelector('.phase span.on') && document.querySelector('.phase span.on').textContent == 'tokens' && "
                               "parseFloat(document.querySelector('.clipprog .bar .fill').style.width) > 40", timeout=180000)
    await pg.screenshot(path=f"{out}/audio-wfe-running.png")
    await pg.wait_for_selector(".song.big", timeout=600000)
    await pg.wait_for_timeout(1500)
    await pg.screenshot(path=f"{out}/audio-wfe-done.png", full_page=True)
    await pg.set_viewport_size({"width": 390, "height": 844})
    await pg.wait_for_timeout(800)
    await pg.screenshot(path=f"{out}/audio-wfe-phone.png", full_page=False)
    await b.close()
    return errors


async def video(p, url, out):
    b = await p.chromium.launch()
    pg = await b.new_page(viewport={"width": 1300, "height": 1100})
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    await pg.goto(url)
    await pg.wait_for_selector("section.panel", timeout=60000)
    await pg.wait_for_timeout(2500)
    await pg.screenshot(path=f"{out}/video-studio.png", full_page=True)
    await b.close()
    return errors


async def main(kind, url, out):
    async with async_playwright() as p:
        errors = await {"image": image, "audio": audio, "video": video}[kind](p, url, out)
    try:
        import glob
        from PIL import Image
        for f in glob.glob(f"{out}/{kind}-*.png"):
            Image.open(f).convert("RGB").save(f[:-4] + ".webp", quality=82, method=6)
    except ImportError:
        print("no Pillow: the PNGs stay")
    print("page errors:", errors or "none")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2], sys.argv[3]))
