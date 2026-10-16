"""Breeze TTS 2 in its own code (github.com/breezeblue-ai/breeze-tts: models/breeze.py, breeze_infer/templates.py), on
the CPU in float32 - the reference for nextsycl's audio engine (audio/breezetts2): a request's prompt, the stages'
tensors and a greedy run's frames, dumped as .npy for `nextsycl audio check`.

    python ref.py CHECKPOINT_DIR --code BREEZE_TTS_DIR --text "..." [--instruction "..."] [--ref-audio r.wav --ref-text "..."]
                  [--cfg 4] [--frames 120] [--sample --seed 42] --out DIR
    python ref.py CHECKPOINT_DIR --code BREEZE_TTS_DIR --decode DIR/codes.npy --out DIR     (the codec's decoder alone)

The prompt is built by upstream's own templates (prepare_inputs: plain, instruction, ref_clone_tata, ref_edit_tata) and
the prompt rows by the model's own _merge_input_ids_with_input_values (the T5Gemma2 text encoder on each text segment,
its projection, the reference's codec frames). The frame loop is the eager path of upstream's streaming runtime
(models/fast_streaming.py iter_audio_chunks with cudagraph/depth_decoder_graph.py _full_loop) written out for the CPU:
the backbone's logits (cond / uncond combined by cfg), the repetition penalty 1.1 over the frames' first codes, the
reserved ids 2048..2050 suppressed, the depth decoder's fifteen codes (cfg too). Greedy by default (a deterministic
run to compare against); --sample draws as upstream (temperature 0.9, top-k 50), torch's generator seeded.

Dumps (DIR): request.json; ids.npy / neg_ids.npy (the prompts' token ids, <|AUDIO|> where a frame goes),
mask.npy / neg_mask.npy (1 at text positions), lens.npy / neg_lens.npy (each text segment's length);
te0.npy (the text encoder's output for the first text segment [n, 1152]); prefill.npy / neg_prefill.npy (the
backbone's input rows [L, 2048]); hidden0.npy (the backbone's last hidden state after the prefill [2048]);
logits.npy [T, 2052] (each frame's first-code logits, guided); depth.npy [T, 15, 2051] (each frame's depth logits,
guided); codes.npy [T, 16]; speech.npy / speech.wav (the codec's decode of codes, 24 kHz); for a reference recording
ref.wav (as read, mono) and ref_codes.npy [F, 16].
"""
import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--code", required=True, help="a checkout of github.com/breezeblue-ai/breeze-tts")
    ap.add_argument("--text", default="")
    ap.add_argument("--instruction", default="")
    ap.add_argument("--ref-audio", default=None)
    ap.add_argument("--ref-text", default=None)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--decode", default=None)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    sys.path.insert(0, a.code)
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    from qwen_tts import Qwen3TTSTokenizer

    codec = Qwen3TTSTokenizer.from_pretrained(os.path.join(a.model, "audio_tokenizer"), device_map="cpu")
    codec.model.float()
    if a.decode:
        codes = torch.from_numpy(np.load(a.decode)).long()
        wavs, sr = codec.decode([{"audio_codes": codes}])
        np.save(os.path.join(a.out, "decoded.npy"), np.asarray(wavs[0], dtype=np.float32))
        print("decoded", len(wavs[0]), "samples")
        return

    from transformers import AutoTokenizer
    from breeze_infer.templates import get_template, prepare_inputs, select_template_name
    from models.breeze import BreezeForConditionalGeneration

    tok = AutoTokenizer.from_pretrained(a.model, fix_mistral_regex=False)
    model = BreezeForConditionalGeneration.from_pretrained(a.model, dtype=torch.float32, attn_implementation="eager")
    model.float().eval()
    cfgm = model.config

    req = {"id": "ref", "text": a.text, "speaker": "S0"}
    if a.instruction.strip():
        req["instruction"] = a.instruction.strip()
    if a.ref_audio:
        wav, sr = sf.read(a.ref_audio, always_2d=True, dtype="float32")
        sf.write(os.path.join(a.out, "ref.wav"), wav.mean(1), sr, subtype="FLOAT")
        req["ref_audio_path"] = os.path.join(a.out, "ref.wav")
        req["ref_text"] = (a.ref_text or "").strip()
    name = select_template_name(req)
    with open(os.path.join(a.out, "request.json"), "w") as f:
        json.dump({"template": name, "text": a.text, "instruction": req.get("instruction", ""), "ref_text": req.get("ref_text", ""),
                   "cfg": a.cfg, "frames": a.frames, "sample": a.sample, "seed": a.seed}, f, ensure_ascii=False)
    torch.manual_seed(a.seed)
    inputs = prepare_inputs(tok, codec, model, [req], get_template(name), guidance_scale=a.cfg, guidance_scale_ref=None, guidance_scale_ins=None)

    def save(n, v):
        np.save(os.path.join(a.out, n), np.ascontiguousarray(v))

    if inputs.get("input_values") is not None:
        save("ref_codes.npy", inputs["input_values"][0].long().numpy())
    branches = [("", inputs["input_ids"], inputs["text_ids_mask"], inputs["text_ids_len"], inputs.get("input_values"))]
    use_cfg = a.cfg != 1.0
    if use_cfg:
        branches.append(("neg_", inputs["cfg_negative_prompt_ids"], inputs["cfg_negative_text_ids_mask"], inputs["cfg_negative_text_ids_len"],
                         inputs.get("cfg_negative_input_values")))

    from transformers.cache_utils import DynamicCache

    with torch.inference_mode():
        caches, hidden, logits = [], [], []
        for prefix, ids, mask, lens, values in branches:
            save(f"{prefix}ids.npy", ids[0].long().numpy())
            save(f"{prefix}mask.npy", mask[0].int().numpy())
            save(f"{prefix}lens.npy", lens.long().numpy())
            merged = model._merge_input_ids_with_input_values(input_ids=ids, input_values=values, text_ids_mask=mask, text_ids_len=lens,
                                                              attention_mask=torch.ones_like(ids))
            emb = merged["inputs_embeds"]
            save(f"{prefix}prefill.npy", emb[0].float().numpy())
            cache = DynamicCache(config=model.backbone_model.config)
            out = model.backbone_model(inputs_embeds=emb, past_key_values=cache, use_cache=True,
                                       position_ids=torch.arange(emb.shape[1]).unsqueeze(0))
            caches.append(cache)
            hidden.append(out.last_hidden_state[:, -1, :])
            logits.append(model.lm_head(out.last_hidden_state[:, -1, :].float()).float())
        # the text encoder on the first text segment alone (as the model runs each segment: its own sequence)
        seg = inputs["input_ids"][0][inputs["text_ids_mask"][0].bool()][: int(inputs["text_ids_len"][0])]
        te = model.text_encoder(input_ids=seg[None], attention_mask=torch.ones_like(seg[None]), position_ids=torch.arange(len(seg))[None])
        save("te0.npy", te.last_hidden_state[0].float().numpy())
        save("hidden0.npy", hidden[0][0].numpy())

        n_cb = cfgm.num_codebooks
        vocab = cfgm.vocab_size  # 2051; the backbone's extra class 2051 is the end
        reserved = list(range(cfgm.codec_config.codebook_size, vocab))
        dd = model.depth_decoder

        def guided(ls):
            return ls[1] + a.cfg * (ls[0] - ls[1]) if use_cfg else ls[0]

        def draw(l, history=None):
            l = l.clone().float()
            if history:
                u = torch.tensor(sorted(set(history)))
                t = l[..., u]
                l[..., u] = torch.where(t > 0, t / 1.1, t * 1.1)
            l[..., reserved] = float("-inf")
            if not a.sample:
                return int(torch.argmax(l, dim=-1))
            l = l / 0.9
            th = torch.topk(l, 50).values[..., -1:]
            l = l.masked_fill(l < th, float("-inf"))
            return int(torch.multinomial(torch.softmax(l, dim=-1), 1))

        all_logits, all_depth, codes, history = [], [], [], []
        pos = [int(c.get_seq_length()) for c in caches]
        lg = guided(logits)
        token = draw(lg[0])
        for step in range(a.frames):
            all_logits.append(lg[0].numpy())
            if token == vocab:
                break
            # the depth decoder: [backbone hidden, c0] then a code a position, each branch its own cache
            dcaches = [DynamicCache(config=dd.config) for _ in caches]
            frame = [token]
            dl = []
            for cb in range(1, n_cb):
                outs = []
                for bi, dc in enumerate(dcaches):
                    if cb == 1:
                        ids2 = torch.tensor([[0, token]])
                        e = dd.model.embed_tokens(ids2 + torch.tensor([[0, 0]]))
                        e[:, 0] = hidden[bi]
                        cp = torch.arange(2)
                    else:
                        e = dd.model.embed_tokens(torch.tensor([[frame[-1] + (cb - 1) * vocab]]))
                        cp = torch.tensor([cb])
                    h = dd.model(inputs_embeds=e, past_key_values=dc, use_cache=True, cache_position=cp).last_hidden_state
                    outs.append(dd.codebooks_head(h[:, -1:, :].float(), cache_position=cp[-1:])[:, 0, :].float())
                g = guided(outs)
                dl.append(g[0].numpy())
                c = draw(g[0])
                frame.append(c)
            all_depth.append(np.stack(dl))
            codes.append(frame)
            history.append(token)
            # the frame fed back to each branch
            fr = torch.tensor([[frame]])
            e = model.backbone_model.embed_tokens(fr)
            hidden, logits = [], []
            for bi, cache in enumerate(caches):
                out = model.backbone_model(inputs_embeds=e, past_key_values=cache, use_cache=True, position_ids=torch.tensor([[pos[bi]]]))
                pos[bi] += 1
                hidden.append(out.last_hidden_state[:, -1, :])
                logits.append(model.lm_head(out.last_hidden_state[:, -1, :].float()).float())
            lg = guided(logits)
            token = draw(lg[0], history)
            print(f"frame {step}: {frame[:4]}...", flush=True)
        save("logits.npy", np.stack(all_logits))
        if all_depth:
            save("depth.npy", np.stack(all_depth))
        codes = np.array(codes, dtype=np.int64).reshape(-1, n_cb)
        save("codes.npy", codes)
        wavs, sr = codec.decode([{"audio_codes": torch.from_numpy(codes)}])
    sf.write(os.path.join(a.out, "speech.wav"), wavs[0], sr)
    save("speech.npy", np.asarray(wavs[0], dtype=np.float32))
    print(f"{name}: {len(wavs[0]) / sr:.2f} s of speech, {codes.shape[0]} frames; prefill {np.load(os.path.join(a.out, 'prefill.npy')).shape}")


if __name__ == "__main__":
    main()
