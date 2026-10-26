#!/usr/bin/env python3
"""Reference test: two film tropes scripted over three clips each, made through the front door's API alone and judged for
consistency across the dimensions docs/video/tropes.md names.

    T14  the mystery object   - a woman finds a music box in an attic (medium), the box opens (extreme close-up), she
                                listens (medium close-up): the face, the box, the room and the light must be the same
                                three times while the shot size changes on purpose
    T30  farewell at the station - she says goodbye (medium close-up), he answers from the carriage door (the reverse),
                                the train leaves with her alone on the platform (wide): two faces and two voices, one
                                station, and the two people looking at each other across the cut

Every clip after the first carries the sound of the one before it (first_audio, its last 22 frames' worth): the room tone, the
music, the idling engine go on across the cut instead of starting again from nothing.

Made as harbour_dialogue.py makes a scene (docs/video/scenes.md S2, S3, S13): pictures are references, not composites - a
place, a character sheet for each person, a prop - and every clip is Ref2VA with the pictures it needs and, for a person
who speaks, that person's voice sample; one speaker a shot; warm light; each line's clip sized to the line.

Then it scores each trope, per dimension:
- measured: the clips' sizes, lengths and sound; for a speaker the median pitch against the voice sample's (as
  harbour_dialogue.py); that the film is the clips cut together;
- judged: a vision model (the chat model, default qwen3.8-27b-q4, temperature 0) is shown frames of the clips - and the
  reference pictures - and asked one question a dimension (the same person? the same box? the same room? the same
  light? did the shot size go as scripted? did they face each other?) for a score from 1 to 5; 4 or more passes. The
  judge is a model, not an oracle: every clip's contact sheet and the frames it saw are written beside the report, so a
  score can be argued with.

    tests/api/trope_scenes.py                         # both tropes, 768x576, 12 steps, upscaled 1.5: ~1 h
    tests/api/trope_scenes.py --only T14 --quick      # one, 384x288, 8 steps, no upscale: ~10 min
    tests/api/trope_scenes.py --no-judge              # the measured checks and the sheets only
    tests/api/trope_scenes.py --url http://box:8000 --token ns-...

Needs: Python 3.9+, an ffmpeg and ffprobe with H.264; on the server minimax-h3 (INT8) and minimax-h3-ref2va, an image model
(qwen-image-2.1-q8), a speech model (qwen3-tts-custom) and, for the judge, a vision chat model (qwen3.8-27b-q4). Writes into
--out/<trope>: pictures, voices, clips, a contact sheet and key frames per clip, film.mp4 and report.json. Exits 1 when a
check fails.
"""

import argparse
import base64
import math
import struct
import wave
import json
import re
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaiju_lighthouse as K   # the API client, the probes, the polling
import harbour_dialogue as H   # the speech, pitch, sheet and cut helpers, and the harbour's people

FPS = 24
WARM_LIGHT = "Live-action, cinematic, warm golden light with soft amber tones, natural warm skin tones, shallow depth of field."
LOCK = "The camera sits in one fixed position on a tripod; that shot size, light level and colour hold from the first frame to the last."
# the same two people as the harbour, in a cardigan and a travelling coat (a different place, so a different costume)
MARA = ("a woman in her forties with short grey-streaked black hair, grey eyes and a kind, weathered face, in a mustard-yellow wool "
        "cardigan over a cream blouse and a rust-red scarf")
ILYA = ("an old man with a full white beard, kind blue eyes and a ruddy face, in a flat tweed cap and a rust-brown wool travelling "
        "coat with a worn leather satchel")
SHEET = H.SHEET


def shot(action, sound, music="None.", look=WARM_LIGHT):
    return (f"integrated_multimodal_description: [Shot 1] {look} {action}\n\noverall_soundscape: {sound}\n\n"
            f"non_diegetic_music: {music}")


def shots(parts, sound, music="None.", look=WARM_LIGHT):
    """A clip of several shots in H3's format: parts = [(time or None, text)]; the first is [Shot 1], the later ones 'At MM:SS.mmm, the camera cuts to ...'"""
    body = []
    for n, (at, text) in enumerate(parts, 1):
        stamp = "" if n == 1 else f"At {int(at // 60):02}:{at % 60:06.3f}, the camera cuts to "
        body.append(f"[Shot {n}] {look if n == 1 else ''} {stamp}{text}".replace("  ", " "))
    return "integrated_multimodal_description: " + "\n".join(body) + f"\n\noverall_soundscape: {sound}\n\nnon_diegetic_music: {music}"


# ---- T14: the mystery object -------------------------------------------------------------------------------------------
ANAT_HANDS = ("Each image is one frame of a film shot. A person has two hands. Count the human hands and forearms visible in each image. Score 5 "
         "if no image shows more than two hands, 1 if any image shows three or more hands or an extra arm that cannot belong to the same person.")
ANAT_BODY = ("Each image is one frame of a film shot. Check each person's body: two arms, two hands, one head, no extra or missing limbs, no second "
             "face. Score 5 if every person in every image is anatomically plausible, 1 if any has an extra limb, a missing limb or a doubled face.")
BOX = "a small antique music box of dark walnut with a mother-of-pearl star inlaid in the lid and brass hinges"
T14 = {
    "title": "the mystery object",
    "pictures": {
        "attic": ("cinematic photograph of a warm dusty attic with a round window, shafts of golden light with floating dust, an open "
                  "wooden trunk and old quilts, honey-coloured beams, no people", "1024x768"),
        "musicbox": (f"photograph of {BOX}, the lid closed, on a plain dark background, even soft warm light", "1024x768"),
        "sheet_mara": (f"{SHEET}: {MARA}", "1344x768"),
    },
    "voices": [],
    "clips": [
        # (id, engine, ref pictures, seconds, prompt)
        ("find", "REF2VA", ["sheet_mara", "attic", "musicbox"], 5.0,
         shot("A medium shot: the woman from the character sheet <Picture 1> kneels in the attic of <Picture 2> beside the open trunk, "
              "lifts out the music box <Picture 3> with both hands and holds it up in the golden light, studying it with curiosity. "
              f"{LOCK}", "Old wood creaking, a soft hush of dust, her breath.")),
        ("open", "REF2VA", ["musicbox", "attic"], 5.0,
         shot("Extreme close-up: a hand holds the music box <Picture 1> in the golden light of <Picture 2>; the lid opens and the brass "
              f"cylinder turns. {LOCK}", "A delicate, slightly out-of-tune music-box melody begins, close and clear, with the small ticking of its gears.")),
        ("listen", "REF2VA", ["sheet_mara", "attic", "musicbox"], 5.0,
         shot("A medium close-up, head and shoulders: the woman from the character sheet <Picture 1> in the attic of <Picture 2> holds "
              "the open music box <Picture 3> close to her chest and listens, her eyes softening and glistening, a small, moved smile. "
              f"{LOCK}", "The same delicate music-box melody continues softly, her slow breath, old wood.")),
    ],
    # (dimension, question, [(clip id or picture, fraction of the clip)]) - the judge scores each from 1 to 5
    "judge": [
        ("D1 identity: the same woman", "Image 1 and image 2 are two shots of a film. Is the woman in image 2 the same person as in image 1 "
         "(same face, hair, clothes)?", [("find", 0.5), ("listen", 0.5)]),
        ("D2 prop: the same music box", "Image 1 is a reference picture of a music box. Images 2, 3 and 4 are shots of a film. Is the music "
         "box in each of images 2, 3 and 4 the same object as in image 1 (same dark wood, same star inlay in the lid, same brass hinges)? "
         "If it cannot be seen in one of them, give 1.", [("@musicbox", 0), ("find", 0.85), ("open", 0.6), ("listen", 0.5)]),
        ("D4 place: the same attic", "Are image 1 and image 2 shots in the same attic (the same beams, light and trunk)?",
         [("find", 0.5), ("listen", 0.5)]),
        ("D5 look: the same light", "Do images 1, 2 and 3 share one lighting and colour palette (warm golden light)? Give 1 if one of them "
         "looks cold or differently graded.", [("find", 0.5), ("open", 0.5), ("listen", 0.5)]),
        ("D8 scale: wide, then macro, then close", "The script says: image 1 a medium shot, image 2 an extreme close-up of the music box, "
         "image 3 a medium close-up of her face. Does the shot size go that way?", [("find", 0.5), ("open", 0.5), ("listen", 0.5)]),
        ("D1 anatomy: the find shot has two hands", ANAT_HANDS, [("find", 0.2), ("find", 0.5), ("find", 0.8)]),
        ("D1 anatomy: the macro has two hands", ANAT_HANDS, [("open", 0.05), ("open", 0.3), ("open", 0.45), ("open", 0.7)]),
        ("D1 anatomy: the close-up has two hands", ANAT_HANDS, [("listen", 0.2), ("listen", 0.6), ("listen", 0.9)]),
        ("D10 performance: she is moved", "Image 1 and image 2 are the start and end of one shot. Does the woman's expression move from "
         "attentive to moved (softer, eyes glistening)?", [("listen", 0.1), ("listen", 0.9)]),
    ],
}

# ---- T30: farewell at the station ---------------------------------------------------------------------------------------
STATION = ("a small old railway station platform in the golden hour, iron lamp posts with warm lamps, honey-coloured brick")
# the geography every shot keeps: the train STANDS at the platform on the right (doors open) until the last shot; she is on the platform
# looking right; he is ON the train, in a carriage doorway, seen from the platform looking left; at the end the same train pulls away to the right
T30 = {
    "title": "farewell at the station",
    "pictures": {
        "platform_a": (f"cinematic photograph of {STATION}, looking along the platform: a green steam train stands motionless at the platform on the "
                       "right with its carriage doors open, no steam rushing, no people", "1024x768"),
        "platform_b": ("cinematic photograph taken from the platform, facing a standing green railway carriage: an open carriage door with a brass "
                       "handrail, the dark carriage interior visible through the doorway, warm lamp light, no people", "1024x768"),
        "train": ("photograph from behind of a vintage green steam train leaving a station platform: the back of the tender and the last carriages "
                  "receding along the track toward a vanishing point, steam rising, brass fittings, golden light, no people", "1024x768"),
        "sheet_mara": (f"{SHEET}: {MARA}", "1344x768"),
        "sheet_ilya": (f"{SHEET}: {ILYA}", "1344x768"),
    },
    "voices": ["mara", "ilya"],
    "clips": [
        ("she", "REF2VA", ["sheet_mara", "platform_a", "@voice_mara"], None,
         ("Write to me when you arrive, and keep warm.", "mara"),
         lambda line: shot(
             "A medium close-up, head and shoulders: the woman from the character sheet <Picture 1> stands on the platform of <Picture 2>; the green "
             "train stands motionless at the platform behind her on the right of the frame, its doors open. She looks to the right of the frame, "
             "at an open carriage door, her eyes glistening, then speaks to the person there in the voice of <Audio 1> (S1): "
             f"<d>[English] {line}</d> The lips move with the words and she smiles bravely at the end. {LOCK}",
             "Her warm voice, close and clear; the quiet of a station, a soft hiss of steam from the standing locomotive.")),
        ("he", "REF2VA", ["sheet_ilya", "platform_b", "@voice_ilya"], None,
         ("I will, and you will wait for me at the harbour.", "ilya"),
         lambda line: shot(
             "A medium close-up, head and shoulders, the reverse shot, seen from the platform: the old man from the character sheet <Picture 1> "
             "stands on the train, in the open doorway of the green carriage of <Picture 2>, one hand on the brass handrail. He looks down to the "
             "left of the frame, at the platform where she stands, then speaks to her in the voice of <Audio 1> (S1): "
             f"<d>[English] {line}</d> The lips move with the words and he nods and smiles at the end. {LOCK}",
             "His warm voice, close and clear; the quiet of a station, a soft hiss of steam from the standing locomotive.")),
        ("away", "REF2VA", ["sheet_mara", "platform_a", "train", "sheet_ilya"], 5.0, (None, None),
         lambda line: shot(
             "A wide shot from the end of the platform of <Picture 2>: the green steam train <Picture 3> pulls away from the camera along the track on "
             "the right, seen from behind, getting smaller and further away, steam billowing; the old man from the character sheet <Picture 4> leans "
             "out of the open door of the last carriage, a small figure waving; the woman from the character sheet <Picture 1> stands on the platform "
             "in the foreground on the left, her back to the camera, waving back. The camera is static.",
             "The locomotive's slow chuffing fading into the distance, steam, the clack of wheels, a distant whistle.")),
    ],
    "judge": [
        ("D1 identity: her", "Image 1 is a character sheet of a woman; images 2 and 3 are two shots of a film. Is the woman in each of "
         "images 2 and 3 the same person as on the sheet (same face, hair, clothes)?", [("@sheet_mara", 0), ("she", 0.5), ("away", 0.2)]),
        ("D1 identity: him", "Image 1 is a character sheet of an old man; image 2 is a shot of a film. Is the man in image 2 the same "
         "person as on the sheet (same beard, cap, coat)?", [("@sheet_ilya", 0), ("he", 0.5)]),
        ("D4 place: the same station", "Are images 1, 2 and 3 shots in the same small railway station (the same lamps, brick and green "
         "steam train)?", [("she", 0.5), ("he", 0.5), ("away", 0.2)]),
        ("D5 look: the same light", "Do images 1, 2 and 3 share one lighting and colour palette (warm golden light)? Give 1 if one of them "
         "looks cold or differently graded.", [("she", 0.5), ("he", 0.5), ("away", 0.2)]),
        ("D7 screen direction: they face each other", "In a shot and reverse shot the two people face opposite directions. In image 1 the "
         "woman should face right and in image 2 the man should face left. Does that hold?", [("she", 0.5), ("he", 0.5)]),
        ("D2 prop: the same train", "Image 1 is a reference picture of a green steam locomotive; image 2 is a shot of a film. Is the "
         "train in image 2 the same locomotive (green, brass fittings, red wheels)?", [("@train", 0), ("away", 0.3)]),
        ("D6 motion: the train leaves", "Images 1, 2 and 3 are the start, middle and end of one shot at a railway platform. A woman stands in the foreground; "
         "a green train is beside the platform. In a farewell the train pulls away along the track. Does the train go away - at the start it is "
         "alongside or near the platform, at the end it is further down the track, smaller, with its rear carriage showing? Give 1 if it stays put or "
         "comes toward the camera.", [("away", 0.05), ("away", 0.5), ("away", 0.95)]),
        ("D4 geography: she speaks to him on the train", "Images 1 to 4 are shots of one farewell at a railway station. Image 1: a woman on the platform "
         "with a green train beside her; image 2: an old man; images 3 and 4: the train leaving. Is the geography legible? In image 1 the train should "
         "STAND at the platform (not arrive); in image 2 the man should stand IN THE DOORWAY OF A TRAIN CARRIAGE (on the train, seen from outside, not "
         "in a room or on the platform); in images 3 and 4 the same train should leave. Give 5 if all three hold, 1 if the train in image 1 is moving "
         "or arriving or the man is not clearly on the train.", [("she", 0.5), ("he", 0.5), ("away", 0.2), ("away", 0.9)]),
        ("D1 anatomy: one body each", ANAT_BODY, [("she", 0.2), ("she", 0.6), ("he", 0.2), ("he", 0.6), ("away", 0.3), ("away", 0.7)]),
        ("D10 performance: the goodbye reads", "Images 1, 2 and 3 are the shots of a farewell. Does the emotion read as a tender goodbye "
         "(her glistening eyes, his kind nod, her alone and waving at the end)?", [("she", 0.7), ("he", 0.7), ("away", 0.8)]),
    ],
}
# ---- T7: the chase -----------------------------------------------------------------------------------------------------
# Hollywood chase grammar, not one long side-on tracking shot: short shots (a cut every 1-2 s) across different set-ups - a low burst out of a doorway, a
# close-up of her panicked face, ground-level boots, an over-the-shoulder, an obstacle thrown behind her, a close-up of his determination, a high angle into the
# alley - the SCREEN DIRECTION constant (every travelling shot goes to the right, he is always behind her), one overcast dusk light, a driving score carried across
PURSUER = ("a tall man in a long black wool coat and a dark grey hat, clean-shaven, with a hard face")
CHASE_LOOK = ("Live-action, cinematic, handheld camera, overcast blue-grey dusk, a wet cobbled street lit by the warm lamps of the shopfronts, steady soft light "
              "with no sun and no glare, shallow depth of field. At a full sprint every stride drives a boot hard into the wet cobbles with a splash, the weight landing, hair and coat bouncing with each step.")
CHASE_SOUND = "Ragged breathing, falling rain, a distant siren."
CHASE_SCORE = "A tense, driving, percussive chase score with low strings and a pounding pulse, building."
T7 = {
    "title": "the chase",
    "max_asl": 3.0,   # the average shot length of the film, s: wide chase shots cut every 2-3 s
    "pictures": {
        # three flat, frontal streets (the runners cross the frame left to right, nobody can approach the camera); A has the crate stack beside a shop door
        "street_a": ("cinematic photograph, a flat frontal view straight across a wet cobbled street at dusk, like a stage set: the camera faces the long row of shopfronts squarely, the pavement and the cobbled road run exactly horizontally from the left edge of the frame to the right edge with no perspective receding into the distance, warm lit shop windows, plain blank painted signboards above the shops with no lettering, puddles reflecting lamplight, a tall stack of wooden crates standing against the wall beside one shop door on the pavement, no people", "1024x768"),
        "street_b": ("cinematic photograph, a flat frontal view straight across a wet cobbled street at dusk, like a stage set: the camera faces the long row of shopfronts squarely, the pavement and the cobbled road run exactly horizontally from the left edge of the frame to the right edge with no perspective receding into the distance, warm lit shop windows, puddles reflecting lamplight, a row of a bakery, a chemist and a bookshop with plain blank painted signboards and no lettering anywhere, no crates, no people", "1024x768"),
        "street_c": ("cinematic photograph, a flat frontal view straight across a wet cobbled street at dusk, like a stage set: the camera faces the long row of shopfronts squarely, the pavement and the cobbled road run exactly horizontally from the left edge of the frame to the right edge with no perspective receding into the distance, warm lit shop windows, puddles reflecting lamplight, a row of a tailor, a cafe and a closed ironmonger with plain blank painted signboards and no lettering anywhere, a dark archway at the right end, no crates, no people", "1024x768"),
        "alley_view": ("cinematic photograph looking down a narrow dark wet alley between brick buildings toward a warm lit gate at the far end, no people", "1024x768"),
        "sheet_mara": (f"{SHEET}: {MARA}", "1344x768"),
        "sheet_pursuer": (f"{SHEET}: {PURSUER}", "1344x768"),
    },
    "voices": [],
    # every shot is wide and static, the runners about a fifth of the frame high, crossing it left to right; each shot is a new street, and a runner who leaves the right
    # edge in one shot is picked up entering the left edge in the next (the cut matches the action)
    "clips": [
        ("break", "REF2VA", ["sheet_mara", "street_b", "sheet_pursuer", "street_c"], 5.0,
         shots([(None, "a wide static shot from across the street of <Picture 2>: the woman from the character sheet <Picture 1> runs flat out along the pavement from the left to the right, "
                       "full body in frame, about a fifth of the frame high, glancing back, and she is ahead on the right; far behind her on the left the tall man in the long black coat from the "
                       "character sheet <Picture 3> runs after her, small, also to the right. She is always ahead of him and he never overtakes her."),
                (2.6, "a wide static shot from across a different street, <Picture 4>: the woman in the yellow coat runs flat out to the right, the same size, full body in frame, ahead on the right; "
                       "the man in the long black coat runs a long way behind her on the left, also to the right. She is ahead, he chases her.")],
               CHASE_SOUND, CHASE_SCORE, look=CHASE_LOOK)),
        ("pursuit", "REF2VA", ["sheet_mara", "street_a", "sheet_pursuer"], 5.0,
         shots([(None, "a wide static shot from across the street of <Picture 2>: the woman from the character sheet <Picture 1> runs along the pavement from the left to the right, "
                       "full body in frame, about a fifth of the frame high, ahead of the man, and as she passes the tall stack of wooden crates standing beside the shop door she hauls it over behind her "
                       "with one arm; the crates crash down across the pavement as she runs on to the right; far behind her on the left the tall man in the black coat runs after her."),
                (2.6, "a wide static shot from across the same street: the crates lie across the pavement; the tall man in the long black coat from the character sheet <Picture 3> runs in from the left edge "
                       "and vaults over them in his stride, the same size, full body in frame, while far ahead of him at the right edge of the frame the small yellow figure of the woman runs away "
                       "to the right. She is ahead, he chases her.")],
               CHASE_SOUND, CHASE_SCORE, look=CHASE_LOOK)),
        ("alley", "REF2VA", ["sheet_mara", "street_c", "sheet_pursuer", "alley_view"], 5.0,
         shots([(None, "a wide static shot from across the street of <Picture 2>: the woman from the character sheet <Picture 1> runs flat out along the pavement from the left to the right, ahead of the man, "
                       "full body in frame, about a fifth of the frame high, and swerves out of frame at the dark archway at the right end; far behind her on the left the man in the black coat runs after her."),
                (2.6, "a wide static shot down the narrow alley of <Picture 4> from its mouth, the warm lamp at the gate lighting the wet cobbles: the woman, small but clearly visible in her "
                       "yellow coat, runs away from the camera toward the lit gate at the far end, and the man in the black coat runs into the alley mouth far behind her. She is ahead, he chases her.")],
               CHASE_SOUND, CHASE_SCORE, look=CHASE_LOOK)),
    ],
    "judge": [
        ("D1 identity: her", "Image 1 is a character sheet of a woman; images 2 and 3 are two shots of a film. Is the woman in each of images 2 and 3 "
         "the same person as on the sheet (same hair, mustard cardigan, rust scarf)? If her face is not visible give your best judgement from hair and clothes.",
         [("@sheet_mara", 0), ("break", 0.8), ("pursuit", 0.5)]),
        ("D1 identity: him", "Image 1 is a character sheet of a man; images 2 and 3 are two shots of a film. Is the man in each of images 2 and 3 the "
         "same person as on the sheet (long black coat, dark grey hat)?", [("@sheet_pursuer", 0), ("pursuit", 0.15), ("pursuit", 0.9)]),
        ("D8 camera distance: wide, full bodies", "Is each of images 1 to 4 a WIDE shot in which a runner is seen full-length (head to feet) with a lot of street around them, "
         "and no close-up or medium shot?", [("break", 0.2), ("break", 0.8), ("pursuit", 0.15), ("alley", 0.15)]),
        ("D7 direction: both run to the right", "In each of images 1 to 4 the people who run along the street should ALL be running toward the RIGHT of the frame. Give 1 if anyone runs "
         "toward the left, or comes toward the camera, or someone crosses the other's path from the other direction.",
         [("break", 0.2), ("break", 0.8), ("pursuit", 0.15), ("pursuit", 0.9)]),
        ("D6 running: a real stride", "In each of images 1 to 3 does a runner have a running stride - legs apart, one foot off the ground, arms pumping - rather than gliding, "
         "hovering or standing?", [("break", 0.8), ("pursuit", 0.15), ("alley", 0.15)]),
        ("D2 obstacle: the crates belong to the street", "Image 1 is the start of a shot and image 2 is a moment later, from one chase. In image 1 a stack of wooden crates should already "
         "stand at the edge of the street beside a shop door, before anyone touches it; in image 2 it has been knocked over. Is the stack present in image 1 at the side of the "
         "street (not in the middle of the roadway), and is it the thing knocked over in image 2? Give 1 if crates appear from nowhere in the middle of the road.",
         [("pursuit", 0.05), ("pursuit", 0.6)]),
        ("D4 order: she leads, he chases", "In each of images 1 to 4 the woman in the yellow coat and the man in the long black coat run to the right. Is the WOMAN always AHEAD (further right) "
         "and the man BEHIND her (to her left), as in a chase where he is after her? Give 1 if in any image the man is ahead of her or she is behind him. If only one of them is visible, judge "
         "by the direction they run and give 3.", [("break", 0.2), ("break", 0.8), ("pursuit", 0.2), ("pursuit", 0.8)]),
        ("D8 text: no garbled lettering", "Is each of images 1 to 3 free of any readable-looking but garbled shop signs, titles or lettering (signboards should be blank or clearly illegible)? "
         "Give 1 if a sign shows strange scrambled letters.", [("break", 0.2), ("pursuit", 0.2), ("alley", 0.2)]),
        ("D4 distance: he is far behind her", "In each of images 1, 2 and 3 a woman in a yellow coat runs along a street and a man in a long black coat chases her. Is the man FAR "
         "behind her - many times her body length - or not in the frame with her at all? Give 1 if he is close behind or beside her.",
         [("break", 0.1), ("pursuit", 0.9), ("alley", 0.45)]),
        ("D5 look: the same light in all three", "Do images 1 and 2 share one lighting and colour grade (a blue-grey dusk with warm shopfront lamps)? Give 1 if one of them has a bright golden sun glare "
         "or is much darker or lighter than the other.", [("break", 0.5), ("pursuit", 0.5)]),
        ("D1 anatomy: one body each", ANAT_BODY, [("break", 0.3), ("break", 0.9), ("pursuit", 0.3), ("pursuit", 0.9), ("alley", 0.3), ("alley", 0.6)]),
    ],
}

# ---- T19: the passage of time -------------------------------------------------------------------------------------------
# the same room at three hours: one base picture, the other two EDITS of it (the layout cannot be rearranged), a static camera; the light and the
# sound change on purpose and everything else holds
ROOM = ("a small cosy room with one tall window on the left, a wooden table with a blue teapot and an open book on it, a red armchair by the wall "
        "on the right, a patterned rug")
NEUTRAL_LOOK = "Live-action, cinematic, natural light, shallow depth of field."
STILL = "The camera is static on a tripod; the room is empty; the furniture and the objects do not move."
T19 = {
    "title": "the passage of time",
    "dissolve": True,   # time passes: the picture dissolves between the hours (a hard cut jumps the background colour), over the sound's crossfade
    "first_frames": {"dawn": "room_dawn", "noon": "room_noon", "dusk": "room_dusk"},
    "pictures": {
        "room_dawn": (f"cinematic photograph of {ROOM}, early dawn light, soft pink and blue, long soft shadows, no people", "1024x768"),
        # the other two hours are COLOUR GRADES of the dawn picture (an img2img re-light at a strength that changed the hour also changed the rendering:
        # crunchy, HDR-like, saturated; at a strength that kept the rendering the hour did not change)
        "room_noon": ("grade:colorbalance=rs=-0.06:bs=0.03:rm=-0.02:bm=0.0:rh=0.04:bh=-0.02,eq=brightness=0.14:contrast=1.2:gamma=1.15:saturation=1.05",
                      "1024x768", "room_dawn"),
        "room_dusk": ("grade:colorbalance=rs=0.16:bs=-0.2:rm=0.12:bm=-0.15:rh=0.08:bh=-0.1,eq=brightness=-0.08:contrast=1.12:saturation=1.2",
                      "1024x768", "room_dawn"),
    },
    "voices": [],
    "clips": [
        ("dawn", "INT8", [], 5.0,
         shot(f"{ROOM[0].upper() + ROOM[1:]}, at dawn: the soft pink and blue light slowly brightens, the light on the wall slowly warms. {STILL}",
              "Birdsong at dawn, a clock ticking softly in the room.", look=NEUTRAL_LOOK)),
        ("noon", "INT8", [], 5.0,
         shot(f"{ROOM[0].upper() + ROOM[1:]}, at midday: strong white-yellow sunlight pours through the window and lies on the floor. {STILL}",
              "A distant street, a fly buzzing at the window, a clock ticking softly in the room.", look=NEUTRAL_LOOK)),
        ("dusk", "INT8", [], 5.0,
         shot(f"{ROOM[0].upper() + ROOM[1:]}, at dusk: the low orange-red light slowly fades to a warm glow, long shadows creep across the rug. {STILL}",
              "Crickets outside, a distant church bell, a clock ticking softly in the room.", look=NEUTRAL_LOOK)),
    ],
    "judge": [
        ("D4 place: the same room", "Are images 1, 2 and 3 the same room: the same tall window on the left, the same table with a blue teapot and an open "
         "book, the same red armchair on the right and rug, all in the same places?", [("dawn", 0.5), ("noon", 0.5), ("dusk", 0.5)]),
        ("D2 props: the teapot and the book stay", "Is the blue teapot and the open book on the table present, the same and in the same place in each of "
         "images 1, 2 and 3?", [("dawn", 0.5), ("noon", 0.5), ("dusk", 0.5)]),
        ("D5 hour: dawn, noon, dusk", "Image 1 should look like dawn (soft pink and blue light), image 2 like bright midday (strong white-yellow sun, a "
         "blue sky outside), image 3 like dusk (low orange-red sunset light). Does each image look like its hour?", [("dawn", 0.5), ("noon", 0.5), ("dusk", 0.5)]),
        ("D5 style: one film", "Images 1, 2 and 3 are the same room at three hours, so IGNORE differences in the brightness and colour of the light. Do the three "
         "have the same sharpness, texture and level of detail? Give 1 only if one looks sharpened, crunchy or HDR-like compared with the others.",
         [("dawn", 0.5), ("noon", 0.5), ("dusk", 0.5)]),
        ("D4 empty: nobody appears", "Is each of images 1, 2 and 3 free of any person, animal or moving figure?", [("dawn", 0.5), ("noon", 0.5), ("dusk", 0.5)]),
    ],
}


def mean_rgb(clip, frac=0.5):
    d = K.probe(clip)["seconds"]
    o = subprocess.run([*K.FFMPEG, "-v", "error", "-ss", f"{d * frac:.3f}", "-i", str(clip), "-frames:v", "1", "-vf", "scale=1:1", "-f", "rawvideo",
                        "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return tuple(o[:3])


def crispness(clip, frac=0.5):
    """Mean absolute grey step between neighbouring pixels of a clip's middle frame (128x96): high for an over-sharpened, HDR-like picture"""
    d = K.probe(clip)["seconds"]
    o = subprocess.run([*K.FFMPEG, "-v", "error", "-ss", f"{d * frac:.3f}", "-i", str(clip), "-frames:v", "1", "-vf", "scale=128:96,format=gray", "-f", "rawvideo", "-"],
                       capture_output=True, check=True).stdout
    W, H = 128, 96
    e = sum(abs(o[y * W + x + 1] - o[y * W + x]) for y in range(H) for x in range(W - 1)) + sum(abs(o[(y + 1) * W + x] - o[y * W + x]) for y in range(H - 1) for x in range(W))
    return e / (W * H)


def frame_change(clip):
    """Mean absolute grey change between consecutive frames of a clip (64x48): near 0 for a static camera on a still room; sparkles and highlights
    that appear and disappear (floating dust, a crawling texture) raise it"""
    o = subprocess.run([*K.FFMPEG, "-v", "error", "-i", str(clip), "-vf", "scale=64:48,format=gray", "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    n = 64 * 48
    fr = [o[i:i + n] for i in range(0, len(o) - n + 1, n)]
    d = [sum(abs(a - b) for a, b in zip(fr[i], fr[i + 1])) / n for i in range(len(fr) - 1)]
    return sum(d) / max(1, len(d))


def colour_steps(film, skip_end=1.0):
    """The largest change in a film's mean picture colour between consecutive frames (sum over R, G, B of an 8x6 mean), and when, leaving out the end fade"""
    o = subprocess.run([*K.FFMPEG, "-v", "error", "-i", str(film), "-vf", "scale=8:6,format=rgb24", "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    n = 8 * 6 * 3
    fr = [o[i:i + n] for i in range(0, len(o) - n + 1, n)]
    m = [[sum(f[c::3]) / 48 for c in range(3)] for f in fr]
    steps = [(sum(abs(a - b) for a, b in zip(m[i], m[i + 1])), (i + 1) / FPS) for i in range(len(m) - 1) if (i + 1) / FPS < len(m) / FPS - skip_end]
    return max(steps)


def shot_cuts(film):
    """Where a film cuts (seconds): consecutive frames (64x48 grey) that differ by more than 10 and four times the median step - the scene-change score of ffmpeg
    missed the cuts in a dark film (it found 2 of 8)"""
    o = subprocess.run([*K.FFMPEG, "-v", "error", "-i", str(film), "-vf", "scale=64:48,format=gray", "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    n = 64 * 48
    fr = [o[i:i + n] for i in range(0, len(o) - n + 1, n)]
    d = [sum(abs(a - b) for a, b in zip(fr[i], fr[i + 1])) / n for i in range(len(fr) - 1)]
    med = sorted(d)[len(d) // 2]
    return [(i + 1) / FPS for i, x in enumerate(d)
            if x > max(10.0, 4 * med) and (i == 0 or d[i - 1] < 0.5 * x) and (i + 1 >= len(d) or d[i + 1] < 0.5 * x)]   # a spike, not a burst of fast motion


def measure_t19(local, ck):
    """The light changes on purpose: noon brightest, dusk warmest, dawn bluest of the warm pair"""
    c = {cid: mean_rgb(dst) for cid, dst, _ in local}
    luma = {k: 0.299 * r + 0.587 * g + 0.114 * b for k, (r, g, b) in c.items()}
    warm = {k: (r + 1) / (b + 1) for k, (r, g, b) in c.items()}
    ck[f"noon is the brightest hour (luma dawn {luma['dawn']:.0f}, noon {luma['noon']:.0f}, dusk {luma['dusk']:.0f})"] = luma["noon"] > luma["dawn"] and luma["noon"] > luma["dusk"]
    fc = {cid: frame_change(dst) for cid, dst, _ in local}
    ck[f"no flicker in the still room (frame-to-frame change {fc['dawn']:.2f}, {fc['noon']:.2f}, {fc['dusk']:.2f}: each under 0.8)"] = max(fc.values()) < 0.8
    cr = {cid: crispness(dst) for cid, dst, _ in local}
    ck[f"the three hours are rendered alike (crispness {cr['dawn']:.1f}, {cr['noon']:.1f}, {cr['dusk']:.1f}: none over 1.6x another)"] = max(cr.values()) <= 1.6 * min(cr.values())
    ck[f"dusk is warmer than noon and dawn (red/blue {warm['dawn']:.2f}, {warm['noon']:.2f}, {warm['dusk']:.2f})"] = warm["dusk"] > warm["noon"] and warm["dusk"] > warm["dawn"]


MEASURES = {"T19": measure_t19}
TROPES = {"T7": T7, "T14": T14, "T19": T19, "T30": T30}


def keyframe(clip, dst, frac, width=640):
    d = K.probe(clip)["seconds"]
    subprocess.run([*K.FFMPEG, "-y", "-v", "error", "-ss", f"{max(0.0, d * frac):.3f}", "-i", str(clip), "-frames:v", "1",
                    "-vf", f"scale={width}:-1", "-q:v", "3", str(dst)], check=True)
    return dst


def data_url(p):
    mime = "image/png" if str(p).endswith(".png") else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(Path(p).read_bytes()).decode()


def ask(api, model, question, images):
    """The judge's score (1-5) and its reason for one question about some pictures"""
    content = []
    for i, p in enumerate(images, 1):
        content += [{"type": "text", "text": f"Image {i}:"}, {"type": "image_url", "image_url": {"url": data_url(p)}}]
    content.append({"type": "text", "text": f"{question}\n\nAnswer with a single JSON object: {{\"score\": <1-5, 5 = clearly yes, 1 = clearly "
                                            "no>, \"why\": \"<one sentence>\"}} and nothing else."})
    body = {"model": model, "temperature": 0, "max_tokens": 2000, "messages": [{"role": "user", "content": content}]}
    r = json.loads(api._req("POST", "/v1/chat/completions", body, 900))
    text = (r.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    m = re.search(r'"score"\s*:\s*([1-5])', text)
    why = re.search(r'"why"\s*:\s*"([^"]*)"', text)
    return (int(m.group(1)) if m else None), (why.group(1) if why else text.strip()[:200])


def norm_words(text):
    return re.findall(r"[a-z']+", text.lower())


def transcribe(api, clip):
    """What a clip says, with times: [(word, start_s, end_s)] from the server's recogniser (whisper)"""
    wav = Path(clip).with_suffix(".wav")
    subprocess.run([*K.FFMPEG, "-y", "-v", "error", "-i", str(clip), "-vn", "-ac", "1", "-ar", "16000", str(wav)], check=True)
    r = api.rpc("audio.transcribe", {"audio": base64.b64encode(wav.read_bytes()).decode(), "timestamps": True}, 900)
    wav.unlink()
    return [(norm_words(w["text"])[0], w["start"], w["end"]) for w in r.get("words", []) if norm_words(w["text"])]


def blend_join(clips, targets, out, speech, fade=1.0, max_overlap=0.5, max_ramp=1.0, dissolve=False):
    """Plain picture cuts, and the sound blended at a cut where no speech sits on it: each clip's sound levelled alone to its LUFS, then
    - a cut with quiet either side (no word within the overlap and the ramp of it): the new clip's sound comes in gradually (a ramp of up to
      `max_ramp` s) and overlaps the old one's end in a crossfade of up to `max_overlap` s; the new clip's picture is trimmed by the overlap so
      sound and picture stay in step (a J-cut)
    - a cut with speech near it: a plain cut, 6 ms in and out so it does not click
    - with `dissolve` (time passing: the same place at another hour) the picture dissolves over the same overlap as the sound crossfades, nothing trimmed
    `speech[i]` is (first_word_start, last_word_end) of clip i in seconds, or None if it says nothing. Returns (seconds, [(overlap, ramp)] a cut)"""
    n, vparts, aparts, blends = len(clips), [], [], [(0.0, 0.0)]
    durs = [K.probe(c)["frames"] / FPS for c in clips]
    for i in range(1, n):
        prev, cur = speech[i - 1], speech[i]
        tail = durs[i - 1] - (prev[1] if prev else 0.0) if prev else durs[i - 1]
        head = cur[0] if cur else durs[i]
        d = min(max_overlap, tail - 0.05, head - 0.05)
        d = round(d * FPS) / FPS if d >= 0.15 else 0.0
        blends.append((d, min(max_ramp, head - 0.05) if d else 0.0))
    for i, (c, t) in enumerate(zip(clips, targets)):
        e = K.loudness(c)
        g = f"volume={t - e:.2f}dB," if e is not None else ""
        d, ramp = blends[i]
        # the sound exactly the picture's length (a clip's sound runs a fraction of a frame long)
        fin = f"afade=t=in:d={ramp:.3f}:curve=qsin" if ramp else "afade=t=in:d=0.006"
        nxt_blends = i + 1 < n and blends[i + 1][0] > 0
        fout = "" if nxt_blends else f",afade=t=out:st={durs[i] - 0.006:.4f}:d=0.006"
        aparts.append(f"[{i}:a]{g}apad,atrim=end={durs[i]:.6f},asetpts=PTS-STARTPTS,alimiter=limit=0.891:level=disabled,{fin}{fout}[a{i}]")
        vparts.append(f"[{i}:v]setpts=PTS-STARTPTS,fps={FPS},format=yuv420p[v{i}]" if dissolve
                      else f"[{i}:v]trim=start_frame={round(d * FPS)},setpts=PTS-STARTPTS[v{i}]")
    chain, last = [], "a0"
    for i in range(1, n):
        d = blends[i][0]
        nxt = f"A{i}"
        if d:
            chain.append(f"[{last}][a{i}]acrossfade=d={d:.4f}:c1=qsin:c2=qsin[{nxt}]")
        else:
            chain.append(f"[{last}][a{i}]concat=n=2:v=0:a=1[{nxt}]")
        last = nxt
    total = sum(durs) - sum(d for d, _ in blends)
    fc = aparts + vparts + chain
    if dissolve:   # picture: a dissolve over each blend's overlap, offset to where the clip before ends less the overlap
        vlen, lastv = durs[0], "v0"
        for i in range(1, n):
            d = blends[i][0]
            fc.append(f"[{lastv}][v{i}]xfade=transition=fade:duration={d:.4f}:offset={vlen - d:.4f}[V{i}]" if d
                      else f"[{lastv}][v{i}]concat=n=2:v=1:a=0[V{i}]")
            vlen += durs[i] - d
            lastv = f"V{i}"
        fc.append(f"[{lastv}]null[v]")
    else:
        fc.append("".join(f"[v{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0[v]")
    fc.append(f"[v]fade=t=out:st={total - fade:.3f}:d={fade}[vf];[{last}]afade=t=out:st={total - fade:.3f}:d={fade}[af]")
    cmd = [*K.FFMPEG, "-y", "-v", "error"]
    for c in clips:
        cmd += ["-i", str(c)]
    cmd += ["-filter_complex", ";".join(fc), "-map", "[vf]", "-map", "[af]", "-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-pix_fmt", "yuv420p", "-r", str(FPS), "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out)]
    subprocess.run(cmd, check=True)
    return total, blends


def level_db(wav_samples):
    return 20 * math.log10(max(1.0, (sum(x * x for x in wav_samples) / max(1, len(wav_samples))) ** 0.5) / 32768)


def cut_steps(film, cuts, window=0.15):
    """The change in sound level across each cut (dB, after minus before, over `window` s) of a film"""
    wav = Path(film).with_suffix(".steps.wav")
    subprocess.run([*K.FFMPEG, "-y", "-v", "error", "-i", str(film), "-vn", "-ac", "1", "-ar", "16000", str(wav)], check=True)
    with wave.open(str(wav)) as w:
        d = struct.unpack("<%dh" % w.getnframes(), w.readframes(w.getnframes()))
    wav.unlink()
    r = 16000
    return [round(level_db(d[int(t * r):int((t + window) * r)]) - level_db(d[int((t - window) * r):int(t * r)]), 1) for t in cuts]


def make(api, a, key, root):
    t = TROPES[key]
    out = root / key
    out.mkdir(parents=True, exist_ok=True)
    w, h = a.width or (384 if a.quick else 768), a.height or (288 if a.quick else 576)
    steps = a.steps or (8 if a.quick else 12)
    upscale = a.upscale if a.upscale is not None else (1.0 if a.quick else 1.5)
    run = time.strftime("%Y%m%d-%H%M%S")
    project = f"{key} {t['title']} {run}"
    report = {"trope": key, "title": t["title"], "project": project, "settings": {"width": w, "height": h, "steps": steps, "upscale": upscale,
              "seed": a.seed}, "clips": {}, "checks": {}, "judge": {}}
    files, pitches = {}, {}

    def up(name, path):
        files[Path(name).stem] = api.rpc("video.upload", {"name": f"{key.lower()}_{name}", "data": base64.b64encode(Path(path).read_bytes()).decode()})["name"]

    for m in ("minimax-h3", "minimax-h3-q6k", "minimax-h3-q4km", "minimax-h3-ref2va"):
        try:
            api.rpc("system.unload", {"model": m})
        except RuntimeError:
            pass
    for p, spec in t["pictures"].items():
        prompt, size, base = spec[0], spec[1], (spec[2] if len(spec) > 2 else None)   # base: a picture of this trope that this one is an edit of (same room, another hour)
        dst = out / f"{p}.png"
        if not dst.exists():
            print(f"[{key}] picture {p}: {a.image_model} {size}{' (re-lit from ' + base + ')' if base else ''}", flush=True)
            if base and prompt.startswith("grade:"):   # the same picture at another hour by a colour grade: the rendering and the layout cannot change
                subprocess.run([*K.FFMPEG, "-y", "-v", "error", "-i", str(out / f"{base}.png"), "-vf", prompt[6:], str(dst)], check=True)
                r = None
            elif base:
                src = "data:image/png;base64," + base64.b64encode((out / f"{base}.png").read_bytes()).decode()
                r = api.rpc("image.generate", {"model": a.image_model, "prompt": prompt, "image": src, "strength": spec[3] if len(spec) > 3 else 0.6,
                                               "size": size, "steps": 28, "seed": a.seed})   # img2img: the base's layout, re-lit, in the base's own rendering
            else:
                r = api.rpc("image.generate", {"model": a.image_model, "prompt": prompt, "size": size, "steps": 28, "seed": a.seed})
            if r:
                url = r["data"][0]["url"]
                api.get(url[url.index("/v1/"):], dst)
        up(f"{p}.png", dst)
    api.rpc("system.unload", {"model": a.image_model})
    for who in t["voices"]:
        dst = out / f"voice_{who}.wav"
        if not dst.exists():
            print(f"[{key}] voice {who}: {a.speech_model}", flush=True)
            r = api.rpc("audio.speak", {"model": a.speech_model, "input": H.VOICE[who][1], "language": "english", "voice": H.VOICE[who][0],
                                        "instructions": H.WARM, "response_format": "url"})
            url = r.get("url") or r["data"][0]["url"]
            api.get(url[url.index("/v1/"):], dst)
        up(f"voice_{who}.wav", dst)
        pitches[who] = H.pitch(dst, 16000)
    if t["voices"]:
        api.rpc("system.unload", {"model": a.speech_model})
        report["voices_hz"] = pitches
        print(f"[{key}] voices: pitch {pitches}", flush=True)

    t0, local, prev_name, prev_spoke, first_nf = time.time(), [], None, False, 0
    G = 22   # frames a guided clip carries
    for i, c in enumerate(t["clips"], 1):
        if len(c) == 5:   # T14: no speech
            cid, engine, refs, secs, prompt = c
            who, line = None, None
        else:             # T30: (id, engine, refs, secs, (line, who), prompt builder)
            cid, engine, refs, secs, (line, who), build = c
            prompt = build(line)
        nf = H.frames_for(line) if line else H.snap((secs or 5.0) * FPS)
        guided = bool(t.get("guided")) and prev_name is not None   # the shot goes on: a later clip begins with the last G frames of the one before
        if guided:
            nf = K.snap(first_nf + G, nearest=True)
        first_nf = first_nf or nf
        body = {"prompt": prompt, "seconds": round(nf / FPS, 4), "steps": steps, "seed": a.seed + i, "width": w, "height": h, "project": project,
                "queue": True, "label": f"{key} {run} {i:02}/{len(t['clips']):02}: {cid}"}
        if [r for r in refs if not r.startswith("@")]:
            body["ref_images"] = [files[r] for r in refs if not r.startswith("@")]
        if cid in t.get("first_frames", {}):   # a clip that starts on a picture (the room at its hour)
            body["first_frame"] = files[t["first_frames"][cid]]
        auds = [files[r[1:]] for r in refs if r.startswith("@voice")]
        if auds:
            body["ref_audios"] = auds
        if guided:   # picture and sound both go on (the join overlaps them)
            body |= {"guide_clip": f"{prev_name}:{G}:0", "first_audio": prev_name, "first_audio_s": round(G / FPS, 4)}
        elif prev_name and not prev_spoke:   # the sound goes on across the cut: the last 22 frames' worth of the clip before (no picture is carried, so a plain cut plays
            # nothing twice) - but never the tail of a speaking clip: its last word comes back as the first 0.5 s of this one (heard twice at the cut)
            body |= {"first_audio": prev_name, "first_audio_s": round(22 / FPS, 4)}
        if upscale > 1:
            body |= {"upscale": upscale, "upscaler": "esrgan-general"}
        t1 = time.time()
        dst = out / f"{i:02}_{cid}.mp4"
        if a.reuse and dst.exists():
            name = dst.name
            print(f"{body['label']}: reusing {dst}", flush=True)
        else:
            api.rpc("video.generate", body)
            print(f"{body['label']}: queued ({nf} frames{', ' + str(len(line.split())) + ' words' if line else ''})", flush=True)
            name = K.wait_clip(api, project, i, body["label"])
            dst = api.get(f"/v1/video/files/{name}", dst)
        prev_name = name if not (a.reuse and name == dst.name) else None
        prev_spoke = bool(who)
        p = K.probe(dst)
        H.sheet(dst, out / f"{i:02}_{cid}.png")
        row = {"file": name, "frames": p["frames"], "asked": nf, "width": p["width"], "height": p["height"], "audio": p["audio"],
               "loudness_lufs": K.loudness(dst), "wall_s": round(time.time() - t1), "who": who, "pitch_hz": H.pitch(dst) if who else None}
        report["clips"][cid] = row
        print(f"{cid}: {name} in {row['wall_s']} s, {p['width']}x{p['height']}, {p['seconds']:.2f} s"
              f"{', pitch ' + str(row['pitch_hz']) + ' Hz' if who else ''}", flush=True)
        local.append((cid, dst, who))

    heard = {}
    if not a.no_transcribe and any(who for _, _, who in local):   # what is said and when: the lines once, and where the sound may be blended at a cut
        for m in ("minimax-h3", "minimax-h3-q6k", "minimax-h3-q4km", "minimax-h3-ref2va"):
            try:
                api.rpc("system.unload", {"model": m})
            except RuntimeError:
                pass
        for cid, dst, who in local:
            heard[cid] = transcribe(api, dst)
        report["heard"] = {cid: " ".join(w for w, _, _ in ws) for cid, ws in heard.items()}
    speech = []
    for cid, dst, who in local:   # (first word start, last word end); unknown (no transcription) = speaking throughout when someone speaks
        ws = heard.get(cid)
        if ws is not None:
            speech.append((ws[0][1], ws[-1][2]) if ws else None)
        else:
            speech.append((0.0, K.probe(dst)["frames"] / FPS) if who else None)
    film = out / "film.mp4"
    gd = bool(t.get("guided"))
    if gd:   # one shot going on: each later clip begins with the last G frames of the one before; the join cuts the picture in their middle and cross-fades the sound
        K.join([d for _, d, _ in local], film, G, FPS, -18.0, 1.0)
        blends = [(0.0, 0.0)] * len(local)
    else:
        total, blends = blend_join([d for _, d, _ in local], [-18.0 if who else -22.0 for _, _, who in local], film, speech,
                                   max_overlap=1.2 if t.get("dissolve") else 0.5, max_ramp=1.2 if t.get("dissolve") else 1.0, dissolve=bool(t.get("dissolve")))
    report["blends"] = [{"cut": i, "overlap_s": d, "ramp_s": r} for i, (d, r) in enumerate(blends) if i]
    fp = K.probe(film)
    ck = report["checks"]
    cl = report["clips"].values()
    ck["clips are the size asked"] = all(c["width"] >= w and c["height"] >= h for c in cl)
    ck["clips are the length asked"] = all(c["frames"] == c["asked"] for c in cl)
    ck["clips have sound"] = all(c["audio"] and c["loudness_lufs"] is not None for c in cl)
    trimmed = G * (len(local) - 1) if gd else sum(round(d * FPS) for d, _ in blends)
    ck["film is the clips cut together"] = abs(fp["frames"] - (sum(c["frames"] for c in cl) - trimmed)) <= 1 and fp["audio"]
    if gd:   # each clip begins with the last one's end (the frame G-1 of a later clip is the last clip's last frame)
        seams = [K.corr(K.thumb(local[i][1], last=True), K.thumb(local[i + 1][1], at=G - 1)) for i in range(len(local) - 1)]
        report["seams"] = [round(x, 3) for x in seams]
        ck["each clip begins with the last one's end (seams " + ", ".join(f"{x:.2f}" for x in seams) + ")"] = all(x > 0.8 for x in seams)
    durs = [K.probe(dst)["frames"] / FPS for _, dst, _ in local]
    start = [0.0]   # where each clip's picture starts in the film (each later clip's picture is trimmed by its blend overlap)
    for j in range(len(local) - 1):
        start.append(start[j] + durs[j] - blends[j][0])
    spoken = [(start[j] - blends[j][0] + sp[0], start[j] - blends[j][0] + sp[1]) if sp else None for j, sp in enumerate(speech)]
    cuts = start[1:] if not gd else []
    steps = cut_steps(film, cuts) if cuts else []
    report["cut_steps_db"] = steps
    for i, st in enumerate(steps, 1):   # a cut with no word within 0.3 s either side must not jump in the sound's level
        near = any(w and w[0] < cuts[i - 1] + 0.3 and w[1] > cuts[i - 1] - 0.3 for w in (spoken[i - 1], spoken[i]))
        if not near:
            ck[f"the sound does not jump at cut {i} (step {st} dB, blend {blends[i][0]} s over {blends[i][1]:.2f} s)"] = abs(st) <= 8.0
    for who, ref in pitches.items():
        mine = [c["pitch_hz"] for c in cl if c["who"] == who]
        ck[f"{who}'s voice held (within 25% of the sample's {ref} Hz)"] = bool(ref) and all(m is not None and abs(m - ref) / ref <= 0.25 for m in mine)
    if len(pitches) == 2:
        (wa, wb) = list(pitches)
        a_hz = [c["pitch_hz"] for c in cl if c["who"] == wa and c["pitch_hz"]]
        b_hz = [c["pitch_hz"] for c in cl if c["who"] == wb and c["pitch_hz"]]
        ck[f"{wa}'s clips are higher than {wb}'s"] = bool(a_hz and b_hz) and min(a_hz) > max(b_hz)

    if t.get("dissolve"):   # time passes: the picture must not jump in colour at the join of the hours (it did at a hard cut: steps 49 and 277)
        big, at_s = colour_steps(film)
        report["colour_step"] = {"largest": round(big, 1), "at_s": round(at_s, 2)}
        ck[f"the picture does not jump in colour (largest frame-to-frame step {big:.1f} at {at_s:.1f} s: under 15)"] = big < 15
    if t.get("max_asl"):   # a film that must cut fast (a chase): the average shot length
        cuts = shot_cuts(film)
        asl = fp["seconds"] / (len(cuts) + 1)
        report["cuts_s"] = [round(c, 2) for c in cuts]
        ck[f"the pace of the genre ({len(cuts)} cuts in {fp['seconds']:.1f} s, average shot {asl:.1f} s: at most {t['max_asl']} s)"] = asl <= t["max_asl"]
    if key in MEASURES:
        MEASURES[key](local, ck)

    # what is said: each line once, and a cut must not replay the end of the line before it (first_audio carries a speaking clip's last word)
    if heard:
        prev = None
        for (cid, dst, who), clip_def in zip(local, t["clips"]):
            line = clip_def[4][0] if len(clip_def) == 6 else None
            ws = heard[cid]
            if line:
                want = norm_words(line)
                got = [w for w, _, _ in ws]
                ck[f"{cid}: the line is said, once (heard {len(got)} words for {len(want)})"] = (
                    sum(w in got for w in want) >= 0.7 * len(want) and len(got) <= len(want) + 2)
            if prev is not None and prev[1]:   # the clip before spoke: its last words must not open this one
                tail = norm_words(prev[1])[-3:]
                early = [w for w, st, _ in ws if st < 0.6]
                ck[f"{cid}: does not replay the end of the line before it"] = not (early and early[0] in tail)
            prev = (cid, line)

    # the judge: frames of the clips and the reference pictures, one question a dimension
    if not a.no_judge:
        for m in ("minimax-h3", "minimax-h3-q6k", "minimax-h3-q4km", "minimax-h3-ref2va"):
            try:
                api.rpc("system.unload", {"model": m})
            except RuntimeError:
                pass
        clip_file = {cid: d for cid, d, _ in local}
        print(f"[{key}] judging with {a.judge_model}", flush=True)
        for dim, question, shots in t["judge"]:
            imgs = []
            for src, frac in shots:
                if src.startswith("@"):
                    imgs.append(out / f"{src[1:]}.png")
                else:
                    imgs.append(keyframe(clip_file[src], out / f"frame_{src}_{int(frac * 100):02}.jpg", frac))
            score, why = ask(api, a.judge_model, question, imgs)
            report["judge"][dim] = {"score": score, "why": why, "frames": [p.name for p in imgs]}
            ck[f"{dim} (judge {score}/5, 4 passes)"] = score is not None and score >= 4
            print(f"  {dim}: {score}/5 - {why}", flush=True)
    report["film"] = {"file": str(film), "seconds": fp["seconds"], "wall_s": round(time.time() - t0)}
    (out / "report.json").write_text(json.dumps(report, indent=1))
    ok = True
    for k, v in ck.items():
        print(f"{'ok  ' if v else 'FAIL'} [{key}] {k}")
        ok &= bool(v)
    print(f"[{key}] film: {film}, {fp['width']}x{fp['height']}, {fp['seconds']:.2f} s; report {out / 'report.json'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--token")
    ap.add_argument("--out", default="trope-scenes")
    ap.add_argument("--only", default="T14,T30", help="the tropes to make (T7 chase, T14 mystery object, T19 passage of time, T30 farewell)")
    ap.add_argument("--quick", action="store_true", help="384x288, 8 steps, no upscale")
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--upscale", type=float)
    ap.add_argument("--seed", type=int, default=1954)
    ap.add_argument("--image-model", default="qwen-image-2.1-q8")
    ap.add_argument("--speech-model", default="qwen3-tts-custom")
    ap.add_argument("--judge-model", default="qwen3.8-27b-q4")
    ap.add_argument("--reuse", action="store_true", help="keep the clips already in --out (a run whose judge failed), and make only the missing ones")
    ap.add_argument("--no-transcribe", action="store_true", help="skip transcribing each clip (the lines said once, no replay at a cut)")
    ap.add_argument("--no-judge", action="store_true", help="the measured checks and the sheets only")
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--ffprobe", default="ffprobe")
    a = ap.parse_args()
    K.FFMPEG, K.FFPROBE = shlex.split(a.ffmpeg), shlex.split(a.ffprobe)
    K.ffmpeg_can()
    keys = [k.strip().upper() for k in a.only.split(",") if k.strip()]
    for k in keys:
        if k not in TROPES:
            sys.exit(f"{k}: the tropes here are {', '.join(TROPES)}")
    api = K.Api(a.url, K.token_of(a.token))
    root = Path(a.out)
    ok = True
    for k in keys:
        ok &= make(api, a, k, root)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
