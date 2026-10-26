---
name: video-scene-scripting
description: Script, render and review a scene of several H3 video clips (dialogue, shot / reverse shot, a trope such as the farewell at the station) so that it reads as ONE scene - legible geography, one voice per shot, no dialogue heard twice at a cut, consistent people and props. Use when writing prompts for more than one clip, when carrying sound or picture across a cut (first_audio, guide_clip, references), or when a joined film "sounds doubled", "makes no sense" or "has the wrong person talking".
---

# Scripting a multi-clip video scene

Each clip is rendered alone, so everything that makes a scene one scene has to be written down and then checked. Every
failure below happened (docs/video/tropes.md has the runs; `tests/api/trope_scenes.py` is the reference script).

## 1. Before any render: the staging sheet

Write one line per fact the viewer needs, then make each shot's prompt state its share of it:

- **Where is everyone and everything, and what is standing or moving?** Name it in every shot that shows it. A train that
  waits in shot 1 and leaves in shot 3 must be *standing, doors open* in shot 1 ("the train arrives" is what the model draws
  for a steaming locomotive behind a person).
- **Who is on which side of which line**: she looks right, he looks left, the train is on the right until it goes. Say
  the side in words in both shots and check it on the frames. "On the left at the start" is ambiguous when two people overlap.
- **Where the camera is**: a reverse shot is seen from where the other person stands. "The old man in the carriage doorway,
  seen from the platform, on the train" - not "in the carriage" (that is an interior, and nobody can tell whom she is talking to).
- **The test**: look only at the middle frame of each clip. Can a stranger say who is speaking to whom, where, and what has
  just happened? If not, the staging is wrong, whatever the judge scores.

## 2. Pictures that stage the shot

- A reference picture's **orientation sets the motion**: a locomotive pictured from the front arrives however the prompt
  says "pulls away"; picture it from behind and write "pulls away from the camera, seen from behind, getting smaller".
- One picture per place per camera position (the platform seen along its length; the carriage door seen from the platform),
  one character sheet per person, one picture per prop. Props keep their kind, not their ornament (a five-pointed star came
  out with six to eight points).
- Edit a person's photo **only in the background** if their face must hold (stage restyling slimmed a face); crop tight.

## 3. Sound at a cut - the doubled-dialogue trap

- `first_audio` makes a clip's sound begin with the previous clip's tail. After a **speaking** clip that tail is her last word
  ("warm.") replayed as the first 0.46 s of his clip: the listener hears it twice. **Never carry sound after a clip that
  speaks.** Carry it only when the tail is music or room tone (a music-box melody then holds -24 to -26 dB instead of
  starting from silence and bursting in); with no picture carried, a plain cut then plays nothing twice.
- One speaker per shot, the same voice sample on every clip of that person, each line sized to its clip with a beat after it.
- Level each clip alone when joining; the film's sound is the clips' sound and nothing else.
- **Blend the sound at a cut where no speech sits on it** (a new place, time or camera): a plain join switches the ambience in one
  step (+16.8 dB into the train shot). Bring the new sound in gradually: ramp it in over up to 1 s and overlap it with the old
  sound's end in a crossfade of up to 0.5 s, trimming the new clip's picture by the overlap so sound and picture stay in step.
  Where a word is near the cut (a line's end or start within the overlap and the ramp), keep a plain cut with 6 ms fades.
  `blend_join` in `tests/api/trope_scenes.py` does it from each clip's word times (steps then 0.1 and 3.8 dB).

## 4. Checks that catch what the eye and the judge miss

Run them on every scene before calling it done (`trope_scenes.py` does the first four):

1. **Transcribe each clip** (`audio.transcribe` with timestamps): each line appears once; no word in the first 0.6 s of a clip
   repeats the last words of the line before it.
2. Pitch per speaker against the voice sample, and speaker A's clips above speaker B's.
3. Clip size, length, sound, loudness; the film is the clips cut together; **the sound level does not step by more than 8 dB at a cut with no word within 0.3 s of it**.
4. A vision-model judge, one question per dimension - and **one for every failure seen before** (hands, direction of travel,
   the geography above). It passed a three-handed take and a train arriving while the prompt said leaving: it only answers
   what it is asked, and is generous.
5. **Look at the contact sheets and listen to the film.** A defect the judge never saw (a red blotch on a coat, an extra
   hand, a replayed word) is found here. Send the film to the user, not only the verdict.

## 5. Prompt hygiene that bit us

- Counts and plurals invite extras ("held in two hands ... the hands lift its lid" drew three); keep hands to wide and medium
  shots and show a macro insert's object alone.
- cfg 1: every word renders, negations included. Say what is there.
- Hold shot size, light and colour in words ("from the first frame to the last"); say the place is empty or extras re-roll.

Related: docs/video/scenes.md (what each cut carries), docs/video/tropes.md (the scripts and their results),
docs/video/h3.md (the writer notes with the numbers), model-cards/guides/minimax-h3.md (the prompt guide).
