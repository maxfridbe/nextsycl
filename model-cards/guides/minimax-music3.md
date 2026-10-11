# Prompting MiniMax Music 3 (songs)

Two texts make a song: a **description** (`instructions`) of the music and the **lyrics** (`input`).

## The description
Genre, mood, tempo (BPM if you know it), instruments, the voice (female/male, its timbre and style), the production:
`An upbeat indie-pop song at 118 BPM, bright jangly guitars, warm bass, a breathy female lead with close harmonies,
clean modern mix.`

## The lyrics
- Structure tags on lines of their own: `[intro]`, `[verse]`, `[pre-chorus]`, `[chorus]`, `[bridge]`,
  `[instrumental]`, `[solo]`, `[outro]`. Text after a tag on the same line is dropped.
- One sung line per line; keep lines singable (6-12 syllables); repeat the chorus where it should come back.
- Empty lyrics: an instrumental.

```
[verse]
Morning light filtering through the pine
Coffee cooling on the window line
[chorus]
Softly the world begins to breathe
```

## Length and steps
`seconds` up to the model's maximum (see `audio.info`); a 3-minute song takes a few minutes on one card. The int8
form (`minimax-music3-int8`) is faster at nearly the same quality.
