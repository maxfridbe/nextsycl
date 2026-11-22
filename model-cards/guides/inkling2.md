# Prompting Inkling 2 Mini

Thinking Machines' Inkling 2 Mini answers OpenAI-style requests (`llm.chat`, or `/v1/chat/completions` for streaming
and tools); its own chat template is applied (the file's `tokenizer.chat_template`).

- **It thinks first**: each answer starts with its reasoning, a message of its own (`reasoning_content` in the
  stream), then the answer. `reasoning_effort` sets how much: `low` (0.2), `medium` (0.7), `high` (0.9, the default),
  `max` (0.99) - lower is faster for simple asks. Give it room (`max_tokens` of a few thousand) for hard problems.
- **Sampling**: the model card recommends temperature 1.0 and top-p 1.0 with no top-k / min-p; temperature 0 gives
  greedy answers (repeatable, a little flatter).
- **System prompt**: the role, the rules and the output format, once, at the start; it precedes the effort line the
  template adds.
- **Tools**: OpenAI `tools` definitions are declared in a system message; the model calls them as JSON
  (`{"name": .., "args": {..}}`), returned as `tool_calls`; send each result back as a `tool` message.
- **Long inputs**: put the material first and the question last. The model holds up to 1M tokens; the cards here
  reserve 256K (UD-Q4_K_XL) or 128K (UD-Q8_K_XL) a conversation on one card.
- **Drafts**: its DSpark drafter (beside the model) proposes up to 5 tokens a step; the answer is the same text it
  would write without it (the same draws), only faster where text is predictable - code ~1.5x, prose about the same.
- **Not here**: pictures (the model reads them; its vision tower is not ported to nextsycl yet).
