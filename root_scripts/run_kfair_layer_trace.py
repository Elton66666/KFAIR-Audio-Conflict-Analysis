import itertools
import json
import os
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from kimia_infer.api.kimia import KimiAudio

ROOT = Path("/root/autodl-tmp/kfair")
OUT = ROOT / "outputs"
PREFIX = os.environ.get("KFAIR_PREFIX", "kfair")
MODEL_PATH = os.environ.get("KFAIR_MODEL_PATH", str(ROOT / "models/Kimi-Audio-7B-Instruct"))
PRED = OUT / f"{PREFIX}_layer_trace.jsonl"
LABELS = ["neutral", "happy", "sad", "angry"]
CHOICES = {"neutral": "A", "happy": "B", "sad": "C", "angry": "D"}
CORE = {("neutral", "angry"), ("happy", "sad")}


def main():
    a = pd.read_csv(OUT / "ravdess_expanded_manifest.csv")
    b = pd.read_csv(OUT / "ravdess_statement2_manifest.csv")
    a["statement"], b["statement"] = 1, 2
    manifest = pd.concat([a, b], ignore_index=True)
    pairs = []
    for (actor, statement), group in manifest.groupby(["actor", "statement"]):
        by = {r.label: r for r in group.itertuples()}
        for la, lb in CORE:
            x, y = by[la], by[lb]
            pairs.append(dict(pair_id=f"a{actor:02d}_s{statement}_{la}_{lb}", actor=actor,
                              statement=statement, label_a=la, label_b=lb,
                              file_a=x.filename, file_b=y.filename))
    model = KimiAudio(model_path=MODEL_PATH, load_detokenizer=False)
    fusion = [m for m in model.alm.modules() if hasattr(m, "vq_adaptor")][0]
    fusion.kfair_branch_mode = "full"
    pm = model.prompt_manager
    prompt = "Classify only the speaker's vocal emotion. Choose exactly one: A=neutral, B=happy, C=sad, D=angry. Reply with only A, B, C, or D."
    ids = {k: pm.text_tokenizer.encode(v, bos=False, eos=False)[0] for k, v in CHOICES.items()}
    cache = {}
    for i, r in enumerate(manifest.itertuples(), 1):
        chats = [{"role":"user","message_type":"text","content":prompt},
                 {"role":"user","message_type":"audio","content":r.audio_path}]
        h = pm.get_prompt(chats, output_type="text")
        audio, text, mask, _, _ = h.to_tensor()
        cache[r.filename] = dict(audio=audio.cpu(), text=text.cpu(), mask=mask.cpu(),
                                 feature=h.continuous_feature[0].cpu())
        if i % 48 == 0: print(f"CACHE {i}/192", flush=True)
    cells = [("AA","file_a","file_a"),("AB","file_a","file_b"),
             ("BA","file_b","file_a"),("BB","file_b","file_b")]
    with PRED.open("w", encoding="utf-8") as out:
        for pi, p in enumerate(pairs, 1):
            for cell, dk, ck in cells:
                d, c = cache[p[dk]], cache[p[ck]]
                feat = c["feature"]
                target = d["feature"].shape[1]
                if feat.shape[1] != target:
                    feat = F.interpolate(feat.transpose(1,2).float(), size=target,
                                         mode="linear", align_corners=False).transpose(1,2).to(feat.dtype)
                audio, text, mask = d["audio"].cuda(), d["text"].cuda(), d["mask"].cuda()
                pos = torch.arange(audio.shape[1], device="cuda").unsqueeze(0).long()
                with torch.inference_mode():
                    o = model.alm(input_ids=audio, text_input_ids=text,
                        whisper_input_feature=[feat.cuda()], is_continuous_mask=mask,
                        position_ids=pos, use_cache=False, output_hidden_states=True, return_dict=True)
                    hs = o.hidden_states[:29]
                    for layer, h in enumerate(hs):
                        z = h[:, -1]
                        if layer < 28: z = model.alm.model.norm(z)
                        logp = torch.log_softmax(model.alm.lm_head(z)[0].float(), dim=-1)
                        scores = {lab: float(logp[tok].cpu()) for lab, tok in ids.items()}
                        rec = {**p, "cell":cell, "layer":layer,
                               "contrast_b_minus_a":scores[p["label_b"]]-scores[p["label_a"]],
                               "prediction":max(scores,key=scores.get), "scores":scores}
                        out.write(json.dumps(rec, ensure_ascii=False)+"\n")
                out.flush()
            if pi % 12 == 0: print(f"PAIRS {pi}/{len(pairs)}", flush=True)
    df = pd.DataFrame(json.loads(x) for x in PRED.read_text().splitlines())
    wide = df.pivot(index=["pair_id","actor","statement","label_a","label_b","layer"], columns="cell", values="contrast_b_minus_a").reset_index()
    wide["ME_C"] = .5*((wide.AB-wide.AA)+(wide.BB-wide.BA))
    wide["ME_D"] = .5*((wide.BA-wide.AA)+(wide.BB-wide.AB))
    wide["interaction"] = wide.BB-wide.BA-wide.AB+wide.AA
    wide.to_csv(OUT/f"{PREFIX}_layer_trace_effects.csv", index=False)
    print(wide.groupby("layer")[["ME_C","ME_D","interaction"]].mean().to_string(), flush=True)


if __name__ == "__main__": main()
