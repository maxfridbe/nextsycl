# Prompting the chat models

The chat models answer OpenAI-style requests: `messages` with `system`, `user` and `assistant` turns
(`llm.chat`, or `/v1/chat/completions` for streaming and tools).

- **System prompt**: the role, the rules and the output format, once, at the start.
- **Be concrete**: the task, the inputs, the form of the answer (a list, JSON with these keys, a diff, at most N
  words). Examples of the wanted output help more than adjectives.
- **Long inputs**: put the material first and the question last.
- **Thinking**: the models reason before answering; give them room (`max_tokens` of a few thousand) for hard
  problems, and ask for the final answer in a fixed form so it is easy to find.
- **Tools**: the models that take tools get OpenAI `tools` definitions; others have them dropped by the switcher.
- The models: GLM-5.3-Flash (both cards, long contexts: 64K, 128K, 256K variants), Qwen3.8-Flash-Next (general,
  Coder for code, Swift for short thinking; "uncensored" variants project out refusals). One loads at a time; a
  request for another swaps it (the switcher answers 409 while the loaded one is in use).
