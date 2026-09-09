import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from kimia_infer.api.kimia import KimiAudio

ROOT = Path("/root/autodl-tmp/kfair")
DATA = ROOT / "data/emis_balanced192"
OUT = ROOT / "outputs"
RAW = OUT / "emis_temporal_holdout_predictions.jsonl"
STATS = OUT / "emis_temporal_holdout_statistics.csv"
LABELS = ["neutral", "happy", "sad", "angry"]
CHOICES = {"neutral": "A", "happy": "B", "sad": "C", "angry": "D"}
MODES = ["full", "no_continuous", "no_discrete", "rolled_continuous"]


def load_manifest():
    rows = []
    for name, conflict in (("emis_balanced144_manifest.jsonl", True), ("emis_aligned48_manifest.jsonl", False)):
        for line in (ROOT / "data" / name).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            row["conflict"] = conflict
            row["audio_path"] = str(DATA / row["filename"])
            rows.append(row)
    frame = pd.DataFrame(rows)
    assert len(frame) == 192
    assert frame.audio_path.map(lambda value: Path(value).exists()).all()
    return frame


def completed_keys():
    if not RAW.exists():
        return set()
    return {(row["filename"], row["mode"]) for row in map(json.loads, RAW.read_text(encoding="utf-8").splitlines())}


def bootstrap(group, column, repetitions=10000):
    clustered = group.groupby(["generator", "text_id"])[column].agg(["sum", "count"])
    totals = clustered["sum"].to_numpy()
    counts = clustered["count"].to_numpy()
    rng = np.random.default_rng(20260903)
    indices = rng.integers(0, len(totals), size=(repetitions, len(totals)))
    samples = totals[indices].sum(axis=1) / counts[indices].sum(axis=1)
    return float(group[column].mean()), float(np.quantile(samples, .025)), float(np.quantile(samples, .975))


def summarize():
    frame = pd.DataFrame(map(json.loads, RAW.read_text(encoding="utf-8").splitlines()))
    frame = frame.drop_duplicates(["filename", "mode"], keep="last")
    assert len(frame) == 192 * len(MODES), len(frame)
    frame["acoustic_correct"] = frame.prediction.eq(frame.acoustic_label)
    frame["semantic_follow"] = frame.prediction.eq(frame.semantic_label)
    frame["other"] = ~frame.acoustic_correct & ~frame.semantic_follow
    rows = []
    for keys, group in frame.groupby(["mode", "conflict", "generator"], dropna=False):
        for metric in ("acoustic_correct", "semantic_follow", "other", "acoustic_margin"):
            mean, low, high = bootstrap(group, metric)
            rows.append({"mode": keys[0], "conflict": keys[1], "generator": keys[2], "metric": metric,
                         "mean": mean, "ci_low": low, "ci_high": high, "n": len(group)})
    for keys, group in frame.groupby(["mode", "conflict"], dropna=False):
        for metric in ("acoustic_correct", "semantic_follow", "other", "acoustic_margin"):
            mean, low, high = bootstrap(group, metric)
            rows.append({"mode": keys[0], "conflict": keys[1], "generator": "ALL", "metric": metric,
                         "mean": mean, "ci_low": low, "ci_high": high, "n": len(group)})
    pd.DataFrame(rows).to_csv(STATS, index=False)


def main():
    manifest = load_manifest()
    done = completed_keys()
    model = KimiAudio(model_path=str(ROOT / "models/Kimi-Audio-7B-Instruct"), load_detokenizer=False)
    fusion = [module for module in model.alm.modules() if hasattr(module, "vq_adaptor")][0]
    prompt = "Classify only the speaker's vocal emotion. Choose exactly one: A=neutral, B=happy, C=sad, D=angry. Reply with only A, B, C, or D."
    token_ids = {label: model.prompt_manager.text_tokenizer.encode(choice, bos=False, eos=False)[0] for label, choice in CHOICES.items()}
    for mode in MODES:
        fusion.kfair_branch_mode = mode
        for index, row in enumerate(manifest.to_dict("records"), 1):
            if (row["filename"], mode) in done:
                continue
            messages = [{"role": "user", "message_type": "text", "content": prompt},
                        {"role": "user", "message_type": "audio", "content": row["audio_path"]}]
            started = time.time()
            item = model.prompt_manager.get_prompt(messages, output_type="text")
            audio_ids, text_ids, mask, _, _ = item.to_tensor()
            audio_ids, text_ids, mask = audio_ids.cuda(), text_ids.cuda(), mask.cuda()
            positions = torch.arange(audio_ids.shape[1], device="cuda").unsqueeze(0).long()
            with torch.inference_mode():
                output = model.alm(input_ids=audio_ids, text_input_ids=text_ids,
                                   whisper_input_feature=[item.continuous_feature[0].cuda()],
                                   is_continuous_mask=mask, position_ids=positions,
                                   use_cache=False, return_dict=True)
            log_probs = torch.log_softmax(output.logits[1][0, -1].float(), dim=-1)
            scores = {label: float(log_probs[token].cpu()) for label, token in token_ids.items()}
            result = {**row, "mode": mode, "prediction": max(scores, key=scores.get), "scores": scores,
                      "acoustic_margin": scores[row["acoustic_label"]] - scores[row["semantic_label"]],
                      "seconds": time.time() - started}
            with RAW.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            if index % 24 == 0:
                print(f"{mode} {index}/192", flush=True)
    summarize()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
