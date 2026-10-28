## Porting brief: SkinTokens / TokenRig (VAST-AI Research). Code MIT, weights MIT

Rigging for `model3d rig MESH --model skintokens`: a mesh in, the same mesh back with a glTF skeleton and skin weights
(UniRig's successor: one autoregressive model writes the skeleton and then the skin as discrete tokens).
Reference code (read only, never executed on hardac1): `https://github.com/VAST-AI-Research/SkinTokens` @ 273b691
(a clone is at the session scratchpad `skintokens/repo`).

**Weights (`https://huggingface.co/VAST-AI/SkinTokens`, rev 79736cad0fd84de384d5eede659b4ebd24effe33):**
- `experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt`: 1,131,603,979 B (TokenRig, GRPO-refined; the recommended one)
- `experiments/skin_vae_2_10_32768/last.ckpt`: 487,311,745 B (the FSQ-CVAE skin tokenizer)
- The Qwen3-0.6B *config* only (`download.py` fetches it into `models/`); the decoder is built from scratch, no Qwen weights.
Lightning ckpts are torch pickles: read them with the repo's own safe readers (`gguf::pth` / `gguf::pthst`), never `torch.load` on hardac1.

**Pipeline (find the exact code in `demo.py`, `src/model/tokenrig.py`, `src/model/skin_vae/**`, `src/tokenizer/**`):**
1. Mesh normalisation and point sampling (`src/data/*`, sampler) - UniRig's `prep.rs` / `voxel.rs` already do the like; reuse.
2. Shape encoder: Michelangelo / `ShapeAsLatentPerceiverEncoder` (UniRig's `encoder.rs`).
3. TokenRig: Qwen3-0.6B-shaped decoder (RoPE, GQA, RMSNorm, SwiGLU - the engine has these in the llm kind) over the shape prefix; one
   interleaved sequence: skeleton tokens, then skin tokens. Grammar-constrained sampling (CPU) as `tokenizer_part.py`/`tokenizer/spec.py`;
   defaults top_k 5, top_p 0.95, temperature 1.0, repetition_penalty 2.0, num_beams 10.
4. SkinTokens: the FSQ-CVAE decoder turns skin tokens back into dense per-vertex weights (`skin_fsq_cvae_model.py`, `FSQ.py`).
5. `--use_postprocess` (voxel skin post-process) and `--use_skeleton` (skin an existing skeleton) are options; the first pass may skip them
   if said so in the guide. The bpy-based I/O (`src/rig_package/**`) is replaced by `crates/mesh` + glTF writing as UniRig did.

**Where:** `model3d/skintokens` (own crate, like `model3d/unirig`; do not edit unirig), arch `skintokens`, `does: [rig]`.
Kernels only in a silo (`kernels/model3d/skintokens/silo`), never in shared kernels. Card `model-cards/model3d/skintokens.yml` (two files above,
sha256 from the downloads) and guide `model-cards/guides/skintokens.md`; docs in `docs/3d.md` (rigging section).
