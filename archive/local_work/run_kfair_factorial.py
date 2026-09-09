import itertools
import json
import os
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from kimia_infer.api.kimia import KimiAudio

ROOT = Path("/root/autodl-tmp/kfair")
OUT = ROOT / "outputs"
PREFIX = os.environ.get("KFAIR_PREFIX", "kfair")
MODEL_PATH = os.environ.get("KFAIR_MODEL_PATH", str(ROOT / "models/Kimi-Audio-7B-Instruct"))
PRED = OUT / f"{PREFIX}_factorial_predictions.jsonl"
PAIR_MANIFEST = OUT / f"{PREFIX}_pair_manifest.csv"
LABELS = ["neutral", "happy", "sad", "angry"]
CHOICES = {"neutral": "A", "happy": "B", "sad": "C", "angry": "D"}


def load_manifest():
    a = pd.read_csv(OUT / "ravdess_expanded_manifest.csv")
    b = pd.read_csv(OUT / "ravdess_statement2_manifest.csv")
    a["statement"], b["statement"] = 1, 2
    df = pd.concat([a, b], ignore_index=True)
    assert len(df) == 192
    return df


def build_pairs(df):
    rows = []
    for (actor, statement), group in df.groupby(["actor", "statement"]):
        by_label = {r.label: r for r in group.itertuples()}
        assert set(by_label) == set(LABELS)
        for label_a, label_b in itertools.combinations(LABELS, 2):
            a, b = by_label[label_a], by_label[label_b]
            rows.append({"pair_id": f"a{actor:02d}_s{statement}_{label_a}_{label_b}", "actor": actor,
                         "statement": statement, "label_a": label_a, "label_b": label_b,
                         "file_a": a.filename, "path_a": a.audio_path,
                         "file_b": b.filename, "path_b": b.audio_path})
    pairs = pd.DataFrame(rows)
    assert len(pairs) == 288
    pairs.to_csv(PAIR_MANIFEST, index=False)
    return pairs


def completed():
    if not PRED.exists():
        return set()
    return {(x["pair_id"], x["cell"]) for x in map(json.loads, PRED.read_text(encoding="utf-8").splitlines())}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    pairs = build_pairs(manifest)
    model = KimiAudio(model_path=MODEL_PATH, load_detokenizer=False)
    fusion = [m for m in model.alm.modules() if hasattr(m, "vq_adaptor")]
    assert len(fusion) == 1
    fusion[0].kfair_branch_mode = "full"
    pm = model.prompt_manager
    prompt = "Classify only the speaker's vocal emotion. Choose exactly one: A=neutral, B=happy, C=sad, D=angry. Reply with only A, B, C, or D."
    choice_ids = {label: pm.text_tokenizer.encode(choice, bos=False, eos=False) for label, choice in CHOICES.items()}
    assert all(len(ids) == 1 for ids in choice_ids.values()), choice_ids

    # Cache every original prompt once. Cross-cells reuse D tokens and swap only C features.
    cache = {}
    for i, row in enumerate(manifest.itertuples(), 1):
        chats = [{"role": "user", "message_type": "text", "content": prompt},
                 {"role": "user", "message_type": "audio", "content": row.audio_path}]
        h = pm.get_prompt(chats, output_type="text")
        audio, text, mask, _, _ = h.to_tensor()
        assert len(h.continuous_feature) == 1
        cache[row.filename] = {"audio": audio.cpu(), "text": text.cpu(), "mask": mask.cpu(),
                               "feature": h.continuous_feature[0].cpu(), "path": row.audio_path, "label": row.label}
        if i % 24 == 0:
            print(f"CACHE {i}/192", flush=True)

    done = completed()
    cells = [("AA", "file_a", "file_a"), ("AB", "file_a", "file_b"),
             ("BA", "file_b", "file_a"), ("BB", "file_b", "file_b")]
    for pi, pair in enumerate(pairs.itertuples(), 1):
        for cell, dkey, ckey in cells:
            if (pair.pair_id, cell) in done:
                continue
            dfile, cfile = getattr(pair, dkey), getattr(pair, ckey)
            d, c = cache[dfile], cache[cfile]
            feat = c["feature"]
            target_len = d["feature"].shape[1]
            if feat.shape[1] != target_len:
                feat = F.interpolate(feat.transpose(1, 2).float(), size=target_len,
                                     mode="linear", align_corners=False).transpose(1, 2).to(feat.dtype)
            audio, text, mask = d["audio"].cuda(), d["text"].cuda(), d["mask"].cuda()
            pos = torch.arange(audio.shape[1], device="cuda").unsqueeze(0).long()
            started = time.time()
            with torch.inference_mode():
                _, logits, _ = model.alm.forward(input_ids=audio, text_input_ids=text,
                    whisper_input_feature=[feat.cuda()], is_continuous_mask=mask,
                    position_ids=pos, past_key_values=None, return_dict=False)
            last = torch.log_softmax(logits[0, -1].float(), dim=-1)
            scores = {label: float(last[ids[0]].cpu()) for label, ids in choice_ids.items()}
            pred = max(scores, key=scores.get)
            result = {"pair_id": pair.pair_id, "actor": pair.actor, "statement": pair.statement,
                      "label_a": pair.label_a, "label_b": pair.label_b, "file_a": pair.file_a,
                      "file_b": pair.file_b, "cell": cell, "discrete_file": dfile,
                      "continuous_file": cfile, "discrete_label": d["label"],
                      "continuous_label": c["label"], "prediction": pred,
                      "scores": scores, "contrast_b_minus_a": scores[pair.label_b] - scores[pair.label_a],
                      "discrete_feature_len": int(d["feature"].shape[1]),
                      "source_continuous_len": int(c["feature"].shape[1]),
                      "seconds": round(time.time() - started, 4)}
            with PRED.open("a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n"); f.flush()
        if pi % 24 == 0:
            print(f"PAIRS {pi}/288", flush=True)

    rows = [json.loads(x) for x in PRED.read_text(encoding="utf-8").splitlines()]
    df = pd.DataFrame(rows)
    wide = df.pivot(index="pair_id", columns="cell", values="contrast_b_minus_a").reset_index()
    meta = pairs[["pair_id", "actor", "statement", "label_a", "label_b"]]
    eff = meta.merge(wide, on="pair_id")
    eff["ME_C"] = 0.5 * ((eff.AB-eff.AA) + (eff.BB-eff.BA))
    eff["ME_D"] = 0.5 * ((eff.BA-eff.AA) + (eff.BB-eff.AB))
    eff["interaction"] = eff.BB-eff.BA-eff.AB+eff.AA
    eff.to_csv(OUT / f"{PREFIX}_factorial_effects.csv", index=False)

    cross = df[df.cell.isin(["AB", "BA"])].copy()
    cross["acoustic_follow"] = ((cross.cell == "AB") & (cross.prediction == cross.label_b)) | ((cross.cell == "BA") & (cross.prediction == cross.label_a))
    cross["discrete_follow"] = ((cross.cell == "AB") & (cross.prediction == cross.label_a)) | ((cross.cell == "BA") & (cross.prediction == cross.label_b))
    summary = {"n_pairs": len(eff), "n_cells": len(df), "ME_C_mean": float(eff.ME_C.mean()),
               "ME_D_mean": float(eff.ME_D.mean()), "interaction_mean": float(eff.interaction.mean()),
               "acoustic_follow_rate": float(cross.acoustic_follow.mean()),
               "discrete_follow_rate": float(cross.discrete_follow.mean()),
               "other_prediction_rate": float((~cross.acoustic_follow & ~cross.discrete_follow).mean())}
    (OUT / f"{PREFIX}_factorial_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
