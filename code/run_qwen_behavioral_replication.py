import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from transformers import (
    Qwen2_5OmniConfig,
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
)


ROOT = Path("/root/autodl-tmp/kfair")
OUT = ROOT / "outputs"
MODEL_PATH = Path(
    "/root/qwen_hf/hub/models--Qwen--Qwen2.5-Omni-7B/"
    "snapshots/ae9e1690543ffd5c0221dc27f79834d0294cba00"
)
RAW = OUT / "qwen_behavioral_predictions.jsonl"
STATS = OUT / "qwen_behavioral_statistics.csv"
SUMMARY = OUT / "qwen_behavioral_summary.json"
LABELS = ["neutral", "happy", "sad", "angry"]
CHOICES = {"neutral": "A", "happy": "B", "sad": "C", "angry": "D"}
PROMPT = (
    "Classify only the speaker's vocal emotion. Choose exactly one: "
    "A=neutral, B=happy, C=sad, D=angry. Reply with only A, B, C, or D."
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_manifest():
    rows = []
    for statement, name in (
        (1, "ravdess_expanded_manifest.csv"),
        (2, "ravdess_statement2_manifest.csv"),
    ):
        frame = pd.read_csv(OUT / name)
        for row in frame.to_dict("records"):
            rows.append(
                {
                    **row,
                    "dataset": "RAVDESS",
                    "statement": statement,
                    "semantic_label": "neutral",
                    "acoustic_label": row["label"],
                    "conflict": row["label"] != "neutral",
                    "generator": "human",
                    "cluster_id": f"actor_{int(row['actor']):02d}",
                }
            )
    emis_dir = ROOT / "data/emis_balanced192"
    for name, conflict in (
        ("emis_balanced144_manifest.jsonl", True),
        ("emis_aligned48_manifest.jsonl", False),
    ):
        for line in (ROOT / "data" / name).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            rows.append(
                {
                    **row,
                    "dataset": "EMIS",
                    "audio_path": str(emis_dir / row["filename"]),
                    "conflict": conflict,
                    "cluster_id": f"{row['generator']}_{int(row['text_id']):02d}",
                }
            )
    frame = pd.DataFrame(rows)
    assert len(frame) == 384, len(frame)
    assert frame.audio_path.map(lambda value: Path(value).exists()).all()
    return frame


def load_model():
    processor = Qwen2_5OmniProcessor.from_pretrained(
        MODEL_PATH, local_files_only=True, use_fast=False
    )
    config = Qwen2_5OmniConfig.from_pretrained(MODEL_PATH, local_files_only=True)
    config.enable_audio_output = False
    # Qwen's subclass loader unconditionally reads spk_dict.pt after loading the
    # model.  That file is only used by the speech-generation talker and, with
    # torch<2.6, Transformers intentionally blocks torch.load on it.  We disable
    # audio output above and call the inherited generic safetensors loader, so
    # neither the talker nor its unrelated speaker dictionary is loaded.
    inherited_loader = super(
        Qwen2_5OmniForConditionalGeneration,
        Qwen2_5OmniForConditionalGeneration,
    ).from_pretrained
    model = inherited_loader(
        MODEL_PATH,
        config=config,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cuda:0",
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, processor


def chat_text(processor):
    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a careful speech emotion classifier."}],
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": "audio.wav"},
                {"type": "text", "text": PROMPT},
            ],
        },
    ]
    return processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )


def read_audio(path, target_rate):
    waveform, rate = torchaudio.load(path)
    waveform = waveform.mean(dim=0)
    if rate != target_rate:
        waveform = torchaudio.functional.resample(waveform, rate, target_rate)
    return waveform.numpy()


def make_inputs(processor, template, path):
    waveform = read_audio(path, processor.feature_extractor.sampling_rate)
    inputs = processor(
        text=[template], audio=[waveform], return_tensors="pt", padding=True
    )
    inputs = inputs.to("cuda")
    for key, value in list(inputs.items()):
        if torch.is_floating_point(value):
            inputs[key] = value.to(torch.bfloat16)
    return inputs


def completed_keys():
    if not RAW.exists():
        return set()
    return {row["dataset"] + ":" + row["filename"] for row in map(json.loads, RAW.read_text(encoding="utf-8").splitlines())}


def cluster_bootstrap(group, column, repetitions=10000):
    clustered = group.groupby("cluster_id")[column].agg(["sum", "count"])
    sums = clustered["sum"].to_numpy(dtype=float)
    counts = clustered["count"].to_numpy(dtype=float)
    if len(sums) == 1:
        value = float(sums[0] / counts[0])
        return value, value, value
    rng = np.random.default_rng(20260915)
    indices = rng.integers(0, len(sums), size=(repetitions, len(sums)))
    samples = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
    return (
        float(group[column].mean()),
        float(np.quantile(samples, 0.025)),
        float(np.quantile(samples, 0.975)),
    )


def summarize():
    frame = pd.DataFrame(map(json.loads, RAW.read_text(encoding="utf-8").splitlines()))
    frame = frame.drop_duplicates(["dataset", "filename"], keep="last")
    frame["acoustic_correct"] = frame.prediction.eq(frame.acoustic_label)
    frame["semantic_follow"] = frame.prediction.eq(frame.semantic_label)
    frame["other"] = ~frame.acoustic_correct & ~frame.semantic_follow
    rows = []
    for keys, group in frame.groupby(["dataset", "conflict", "generator"], dropna=False):
        for metric in (
            "acoustic_correct",
            "semantic_follow",
            "other",
            "acoustic_margin",
            "seconds",
        ):
            mean, low, high = cluster_bootstrap(group, metric)
            rows.append(
                {
                    "dataset": keys[0],
                    "conflict": bool(keys[1]),
                    "generator": keys[2],
                    "metric": metric,
                    "mean": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "n": len(group),
                }
            )
    for keys, group in frame.groupby(["dataset", "conflict"], dropna=False):
        for metric in (
            "acoustic_correct",
            "semantic_follow",
            "other",
            "acoustic_margin",
            "seconds",
        ):
            mean, low, high = cluster_bootstrap(group, metric)
            rows.append(
                {
                    "dataset": keys[0],
                    "conflict": bool(keys[1]),
                    "generator": "ALL",
                    "metric": metric,
                    "mean": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "n": len(group),
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(STATS, index=False)
    summary = {
        "model": "Qwen/Qwen2.5-Omni-7B",
        "revision": "ae9e1690543ffd5c0221dc27f79834d0294cba00",
        "architecture_note": (
            "Qwen uses one continuous audio encoder rather than Kimi's separable native "
            "discrete/continuous audio streams. This is a behavioral semantic-by-acoustic "
            "replication, not an internal stream-swap replication."
        ),
        "prompt": PROMPT,
        "n": int(len(frame)),
        "ravdess_accuracy": float(frame[frame.dataset.eq("RAVDESS")].acoustic_correct.mean()),
        "emis_aligned_accuracy": float(
            frame[frame.dataset.eq("EMIS") & ~frame.conflict].acoustic_correct.mean()
        ),
        "emis_conflict_acoustic_follow": float(
            frame[frame.dataset.eq("EMIS") & frame.conflict].acoustic_correct.mean()
        ),
        "emis_conflict_semantic_follow": float(
            frame[frame.dataset.eq("EMIS") & frame.conflict].semantic_follow.mean()
        ),
    }
    SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def main():
    args = parse_args()
    if args.overwrite and RAW.exists():
        RAW.unlink()
    manifest = load_manifest()
    if args.limit:
        manifest = manifest.groupby("dataset", group_keys=False).head(args.limit)
    done = completed_keys()
    model, processor = load_model()
    template = chat_text(processor)
    token_ids = {label: processor.tokenizer.encode(choice, add_special_tokens=False)[0] for label, choice in CHOICES.items()}
    assert all(len(processor.tokenizer.encode(choice, add_special_tokens=False)) == 1 for choice in CHOICES.values())
    for index, row in enumerate(manifest.to_dict("records"), 1):
        key = row["dataset"] + ":" + row["filename"]
        if key in done:
            continue
        started = time.time()
        inputs = make_inputs(processor, template, row["audio_path"])
        with torch.inference_mode():
            output = model.thinker(**inputs, use_cache=False, return_dict=True)
        log_probs = torch.log_softmax(output.logits[0, -1].float(), dim=-1)
        scores = {label: float(log_probs[token].cpu()) for label, token in token_ids.items()}
        result = {
            **row,
            "prediction": max(scores, key=scores.get),
            "scores": scores,
            "acoustic_margin": scores[row["acoustic_label"]] - scores[row["semantic_label"]],
            "seconds": time.time() - started,
        }
        with RAW.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        if index % 16 == 0 or index == len(manifest):
            print(f"BEHAVIOR {index}/{len(manifest)} {row['dataset']}", flush=True)
    if not args.limit:
        summarize()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
