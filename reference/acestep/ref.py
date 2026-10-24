"""ACE-Step 1.5's own pipeline, run as the platform's music worker runs it, with every stage the port checks dumped.

    ACESTEP_CHECKPOINTS_DIR=~/music-maker/models/Ace-Step1.5 PYTHONPATH=~/music-maker/ace-step \
      ~/music-maker/venv/bin/python reference/acestep/ref.py --out ~/acestep-ref/t2m --device cpu \
        --caption "..." --lyrics-file lyrics.txt --duration 30 --seed 42 [--thinking] [--task cover --src a.wav]

Offline diagnosis only (Python is never at runtime): it builds AceStepHandler / LLMHandler the way the platform's
engine does (python/cp_worker/kinds/music/engines/acestep.py, with the same transformers-5 shims), hands
`generate_music` the GenerationParams that engine builds, and wraps the model's own functions to save:

    request.json         the request and the GenerationParams
    text.json            each text the DiT's tokenizer was given and its ids (caption prompt, lyrics)
    te_out.npy           the text encoder's last hidden state of the caption prompt   [L, 1024]
    lyric_emb.npy        the lyrics' token embeddings                                  [Ll, 1024]
    refer.npy            the timbre encoder's input (packed reference latents)        [N, 750, 64]
    src.npy, chunk.npy   src_latents and chunk_masks                                  [T, 64]
    hints.npy            precomputed LM hints (25 Hz) when the LM made codes          [T, 64]
    enc.npy, ctx.npy     prepare_condition's encoder_hidden_states and context_latents
    noise.npy            the starting noise                                            [T, 64]
    step_<i>_x.npy, step_<i>_v.npy, steps.json   each decoder call's input and velocity, and its t
    latents.npy          the sampler's output                                          [T, 64]
    vae_in.npy, wav_raw.npy, audio.npy   the VAE decode's input, its output, the final audio [2, S] (48 kHz)
    lm.json              the planner's prompt(s), its metadata and audio codes (with --thinking)
    vae_enc_in.npy, vae_enc_out.npy      a cover / repaint's source encode
"""

import argparse
import json
import os

import numpy as np
import torch

OUT = None
SAVED = {}


def save(name, t):
    if t is None:
        return
    a = np.ascontiguousarray(t.detach().float().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t))
    np.save(os.path.join(OUT, name), a)
    SAVED[name] = list(a.shape)


def jdump(name, obj):
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f, indent=1, default=str)


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--caption", default="upbeat indie pop, bright electric guitars, punchy drums, warm bass, female lead vocal, summery")
    ap.add_argument("--lyrics", default="")
    ap.add_argument("--lyrics-file", default="")
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--shift", type=float, default=3.0)
    ap.add_argument("--bpm", type=int, default=0)
    ap.add_argument("--key", default="")
    ap.add_argument("--timesig", default="")
    ap.add_argument("--language", default="unknown")
    ap.add_argument("--thinking", action="store_true", help="the planner LM (the platform's use_lm)")
    ap.add_argument("--task", default="text2music")
    ap.add_argument("--src", default="", help="cover / repaint source audio")
    ap.add_argument("--cover-strength", type=float, default=0.5)
    ap.add_argument("--repaint-start", type=float, default=0.0)
    ap.add_argument("--repaint-end", type=float, default=-1.0)
    ap.add_argument("--source-lyrics-file", default="", help="edit: the words the source sings now")
    ap.add_argument("--vae-mean", action="store_true", help="the source's VAE posterior's mean, not a sample (the port's vae-sample 0: exact)")
    ap.add_argument("--checkpoint", default="acestep-v15-turbo")
    ap.add_argument("--max-steps-saved", type=int, default=99)
    ap.add_argument("--platform-engine", default=os.path.expanduser("~/creative-platform/current/python"),
                    help="where cp_worker lives (for ace_compat)")
    args = ap.parse_args()
    if args.src:
        args.src = os.path.abspath(os.path.expanduser(args.src))
    OUT = os.path.expanduser(args.out)
    os.makedirs(OUT, exist_ok=True)
    # stripped, as the platform's engine strips them
    lyrics = (open(args.lyrics_file).read() if args.lyrics_file else args.lyrics).strip()

    # the platform's shims (upstream code is never edited), by path: its engines folder has an acestep.py of its own
    import importlib.util

    spec = importlib.util.spec_from_file_location("ace_compat", os.path.join(os.path.expanduser(args.platform_engine), "cp_worker", "kinds", "music",
                                                                            "engines", "ace_compat.py"))
    ace_compat = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ace_compat)

    ace_compat.apply()
    torch.manual_seed(0)
    if args.vae_mean:
        from diffusers.models.autoencoders.autoencoder_oobleck import OobleckDiagonalGaussianDistribution

        OobleckDiagonalGaussianDistribution.sample = lambda self, generator=None: self.mean
    from acestep.handler import AceStepHandler
    from acestep.inference import GenerationConfig, GenerationParams, generate_music
    from acestep.llm_inference import LLMHandler

    ckpt = os.environ["ACESTEP_CHECKPOINTS_DIR"]
    dit = AceStepHandler()
    msg, ok = dit.initialize_service(project_root=ckpt, config_path=args.checkpoint, device=args.device,
                                     offload_to_cpu=False, quantization=None)
    if not ok:
        raise SystemExit(f"DiT init failed: {msg}")
    print("repair", ace_compat.repair_model(dit.model), "dtype", dit.dtype, flush=True)
    model = dit.model

    # -- the hooks --------------------------------------------------------------------------------------------
    texts = []
    tok = dit.text_tokenizer
    orig_tok_call = type(tok).__call__

    def tok_call(self, text, *a, **kw):
        r = orig_tok_call(self, text, *a, **kw)
        if self is tok and isinstance(text, str):
            ids = r["input_ids"]
            ids = ids[0].tolist() if hasattr(ids, "tolist") and getattr(ids, "ndim", 1) == 2 else (ids.tolist() if hasattr(ids, "tolist") else ids)
            texts.append({"text": text, "ids": ids, "kw": {k: str(v) for k, v in kw.items()}})
        return r

    type(tok).__call__ = tok_call

    orig_gen = model.generate_audio

    def gen(**kw):
        info = {}
        for k, v in kw.items():
            if isinstance(v, torch.Tensor):
                info[k] = list(v.shape)
            else:
                info[k] = v if isinstance(v, (int, float, str, bool, type(None), list)) else str(v)
        jdump("generate_kwargs.json", info)
        save("te_out.npy", kw["text_hidden_states"][0])
        save("text_mask.npy", kw["text_attention_mask"][0])
        save("lyric_emb.npy", kw["lyric_hidden_states"][0])
        save("lyric_mask.npy", kw["lyric_attention_mask"][0])
        save("refer.npy", kw["refer_audio_acoustic_hidden_states_packed"])
        save("src.npy", kw["src_latents"][0])
        save("chunk.npy", kw["chunk_masks"][0])
        save("is_covers.npy", kw["is_covers"])
        if kw.get("precomputed_lm_hints_25Hz") is not None:
            save("hints.npy", kw["precomputed_lm_hints_25Hz"][0])
        if kw.get("non_cover_text_hidden_states") is not None:
            save("te_out_nc.npy", kw["non_cover_text_hidden_states"][0])
        if kw.get("repaint_mask") is not None:
            save("repaint_mask.npy", kw["repaint_mask"][0])
        if kw.get("clean_src_latents") is not None:
            save("clean_src.npy", kw["clean_src_latents"][0])
        if kw.get("silence_latent") is not None:
            save("silence.npy", kw["silence_latent"][0])
        out = orig_gen(**kw)
        save("latents.npy", out["target_latents"][0])
        return out

    model.generate_audio = gen

    calls = {"prep": 0}
    orig_prep = model.prepare_condition

    def prep(*a, **kw):
        r = orig_prep(*a, **kw)
        n = calls["prep"]
        calls["prep"] += 1
        sfx = "" if n == 0 else f"_{n}"
        save(f"enc{sfx}.npy", r[0][0])
        save(f"enc_mask{sfx}.npy", r[1][0])
        save(f"ctx{sfx}.npy", r[2][0])
        return r

    model.prepare_condition = prep

    orig_tokenize = model.tokenize

    def tokenize(*a, **kw):
        r = orig_tokenize(*a, **kw)
        save("codes_src.npy", r[1].reshape(-1))
        return r

    model.tokenize = tokenize

    orig_noise = model.prepare_noise

    def noise(*a, **kw):
        r = orig_noise(*a, **kw)
        save("noise.npy", r[0])
        return r

    model.prepare_noise = noise

    steps = []
    orig_dec = model.decoder.forward

    def dec(*a, **kw):
        r = orig_dec(*a, **kw)
        i = len(steps)
        steps.append({"t": float(kw["timestep"][0]), "r": float(kw["timestep_r"][0])})
        if i < args.max_steps_saved:
            save(f"step_{i}_x.npy", kw["hidden_states"][0])
            save(f"step_{i}_v.npy", r[0][0])
        return r

    model.decoder.forward = dec

    vae = dit.vae
    orig_dec_vae = vae.decode
    orig_enc_vae = vae.encode
    vcalls = {"d": 0, "e": 0}

    def vdec(z, *a, **kw):
        r = orig_dec_vae(z, *a, **kw)
        if vcalls["d"] == 0:
            save("vae_in.npy", z[0])
            save("wav_raw.npy", (r.sample if hasattr(r, "sample") else r[0])[0])
        vcalls["d"] += 1
        return r

    def venc(x, *a, **kw):
        r = orig_enc_vae(x, *a, **kw)
        if vcalls["e"] == 0:
            save("vae_enc_in.npy", x[0])
            lat = r.latent_dist.mean if hasattr(r, "latent_dist") else r[0]
            save("vae_enc_out.npy", lat[0])
        vcalls["e"] += 1
        return r

    vae.decode = vdec
    vae.encode = venc

    # -- the planner ---------------------------------------------------------------------------------------------
    llm = LLMHandler()
    lm_log = {}
    if args.thinking:
        msg, ok = llm.initialize(checkpoint_dir=ckpt, lm_model_path="acestep-5Hz-lm-1.7B", backend="pt",
                                 device=args.device, offload_to_cpu=False)
        if not ok:
            raise SystemExit(f"LM init failed: {msg}")
        orig_lm = llm.generate_with_stop_condition

        def lm(**kw):
            r = orig_lm(**kw)
            lm_log["kwargs"] = {k: v for k, v in kw.items() if k != "progress"}
            lm_log["result"] = {k: v for k, v in r.items() if k != "extra_outputs"}
            lm_log["extra"] = str(r.get("extra_outputs"))[:4000]
            return r

        llm.generate_with_stop_condition = lm
        # each phase: its prompts and their ids, the tokens it sampled, the model's logits and the processed scores of
        # its first steps (lm_<phase>_logits_<i>.npy [rows, vocab]: cond first; lm_<phase>_scores_<i>.npy)
        lm_log["phases"] = []
        cur = {}
        orig_single = llm._run_pt_single

        def single(**kw):
            ph = {"phase": kw.get("generation_phase"), "prompt": kw["formatted_prompt"],
                  "ids": llm.llm_tokenizer(kw["formatted_prompt"])["input_ids"], "tokens": [], "forward": 0, "sample": 0,
                  "cfg_scale": kw.get("cfg_scale"), "temperature": kw.get("temperature"), "top_p": kw.get("top_p"),
                  "target_duration": kw.get("target_duration"), "user_metadata": kw.get("user_metadata")}
            lm_log["phases"].append(ph)
            cur["p"] = ph
            out = orig_single(**kw)
            ph["output"] = out
            return out

        llm._run_pt_single = single
        orig_uncond = llm._build_unconditional_prompt

        def uncond(*a, **kw):
            r = orig_uncond(*a, **kw)
            cur["p"]["uncond_prompt"] = r
            cur["p"]["uncond_ids"] = llm.llm_tokenizer(r)["input_ids"]
            return r

        llm._build_unconditional_prompt = uncond
        orig_fwd = llm._forward_pass

        def fwd(*a, **kw):
            out = orig_fwd(*a, **kw)
            ph = cur["p"]
            if ph["forward"] < 3:
                save(f"lm_{ph['phase']}_logits_{ph['forward']}.npy", out.logits[:, -1, :])
            ph["forward"] += 1
            return out

        llm._forward_pass = fwd
        orig_sample = llm._sample_tokens

        def sample(logits, temperature):
            ph = cur["p"]
            if ph["sample"] < 3:
                save(f"lm_{ph['phase']}_scores_{ph['sample']}.npy", logits)
            ph["sample"] += 1
            t = orig_sample(logits, temperature)
            ph["tokens"].append(int(t[0]))
            return t

        llm._sample_tokens = sample

    # -- the request, as the platform's engine builds it -----------------------------------------------------------
    task = args.task
    if task == "edit":
        # flow-edit: each step's forward noise, and the loop's result
        import acestep.models.common.flow_edit as fe

        orig_draw = fe.draw_fwd_noise
        draws = {"n": 0}

        def draw(*a, **kw):
            r = orig_draw(*a, **kw)
            save(f"edit_noise_{draws['n']}.npy", r[0])
            draws["n"] += 1
            return r

        fe.draw_fwd_noise = draw
        orig_fe = model.flowedit_generate_audio

        def fe_gen(**kw):
            save("src.npy", kw["src_latents"][0])
            out = orig_fe(**kw)
            save("latents.npy", out["target_latents"][0])
            return out

        model.flowedit_generate_audio = fe_gen
    a2a = task in ("cover", "repaint")
    instrumental = not lyrics.strip()
    params = GenerationParams(
        task_type=task,
        caption=args.caption[:512],
        lyrics=lyrics[:4096] if lyrics else "[Instrumental]",
        instrumental=instrumental,
        vocal_language=args.language,
        bpm=args.bpm or None,
        keyscale=args.key,
        timesignature=args.timesig,
        duration=-1.0 if a2a else float(args.duration),
        inference_steps=args.steps,
        guidance_scale=7.0,
        shift=args.shift,
        infer_method="ode",
        thinking=args.thinking,
        lm_temperature=0.85,
        lm_cfg_scale=2.0,
        src_audio=args.src if a2a else None,
        audio_cover_strength=args.cover_strength,
        cover_noise_strength=0.0,
        repainting_start=args.repaint_start,
        repainting_end=args.repaint_end,
        repaint_mode="balanced",
        repaint_strength=0.5,
        chunk_mask_mode="explicit" if task == "repaint" else "auto",
    )
    if task == "lego":
        # the platform's lego_request: the vocals track over the whole recording, base's 50 steps and real guidance
        params = GenerationParams(task_type="lego", instruction="Generate the VOCALS track based on the audio context:", src_audio=args.src,
                                  caption=args.caption[:512], lyrics=lyrics[:4096] if lyrics else "[Verse]", instrumental=False,
                                  vocal_language=args.language, bpm=args.bpm or None, keyscale=args.key, timesignature=args.timesig, duration=-1.0,
                                  inference_steps=args.steps, guidance_scale=7.0, shift=args.shift, infer_method="ode", thinking=False,
                                  repainting_start=0.0, repainting_end=-1.0)
    if task == "edit":
        # the platform's _edit_params: the same caption on both sides, the old words the source's
        import soundfile as sf

        before = open(args.source_lyrics_file).read().strip()
        params = GenerationParams(task_type="text2music", src_audio=args.src, flow_edit_morph=True, caption=args.caption[:512],
                                  flow_edit_source_caption=args.caption[:512], lyrics=lyrics[:4096], flow_edit_source_lyrics=before[:4096],
                                  flow_edit_n_min=0.0, flow_edit_n_max=1.0, flow_edit_n_avg=1, duration=float(sf.info(args.src).duration),
                                  instrumental=False, vocal_language=args.language, inference_steps=args.steps, guidance_scale=7.0,
                                  shift=args.shift, infer_method="ode", thinking=False)
    config = GenerationConfig(batch_size=1, use_random_seed=False, seeds=[args.seed % 2**31], audio_format="flac")
    jdump("request.json", {"args": vars(args), "params": params.to_dict()})
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        res = generate_music(dit, llm, params, config, save_dir=tmp)
    if not res.success or not res.audios:
        raise SystemExit(f"generation failed: {res.error or res.status_message}")
    a = res.audios[0]
    save("audio.npy", a["tensor"])
    jdump("text.json", texts)
    jdump("steps.json", steps)
    if lm_log:
        jdump("lm.json", lm_log)
    jdump("result.json", {"params": a.get("params"), "sample_rate": a.get("sample_rate"), "saved": SAVED})
    import soundfile as sf

    sf.write(os.path.join(OUT, "audio.wav"), a["tensor"].float().cpu().numpy().T, int(a.get("sample_rate", 48000)))
    print(json.dumps({"saved": SAVED, "steps": len(steps)}), flush=True)


if __name__ == "__main__":
    main()
