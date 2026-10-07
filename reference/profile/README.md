# Profiles

`NS_PROFILE=gpu nextsycl generate ...` prints a `[profile ...]` line per section (device timestamps, no syncs).
`tree.py PROMPT_PROFILE DECODE_PROFILE` nests them (`KDA: scan` under KDA, `MoE f16: ...` under the routed experts,
`MLA att: ...` under MLA's attention) into the JSON a flame graph reads; the wall times are the run's own
(`[prompt ... in S s ...]`, set in the script).

The 2026-10-07 runs (B65 + B70, the B70 last): a 36,585-token prompt (455.7 tok/s) and 300 decode tokens (17.8 tok/s,
MTP on). Reading: experts 45% (half of it expanding the 2-bit weights to fp16), MLA 18% (attention at 36K), KDA 18%
(the scan 6.9 s). Decode: experts 50% - mostly waiting for missed experts over PCIe - KDA 19%, MLA 9%.
