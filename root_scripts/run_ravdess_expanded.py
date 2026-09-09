import json
import re
import time
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from kimia_infer.api.kimia import KimiAudio


ROOT = Path("/root/autodl-tmp/kfair")
DATA = ROOT / "data/ravdess_subset/data"
AUDIO_DIR = ROOT / "data/ravdess_eval_audio"
OUT_DIR = ROOT / "outputs"
PRED_PATH = OUT_DIR / "ravdess_expanded_predictions.jsonl"
SUMMARY_PATH = OUT_DIR / "ravdess_expanded_summary.json"
MANIFEST_PATH = OUT_DIR / "ravdess_expanded_manifest.csv"
LABELS = ["neutral", "happy", "sad", "angry"]
MODES = ["full", "no_continuous", "no_discrete", "rolled_continuous"]


def prepare_manifest():
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = [pd.read_parquet(p) for p in sorted(DATA.glob("*.parquet"))]
    df = pd.concat(frames, ignore_index=True)
    # Five actors, identical statement/repetition, normal intensity: 20 paired items.
    chosen = []
    for _, row in df.iterrows():
        parts = row.filename.removesuffix(".wav").split("-")
        if len(parts) != 7:
            continue
        _, _, emotion, intensity, statement, repetition, actor = parts
        if int(actor) <= 24 and statement == "01" and repetition == "01" and intensity == "01" and row["style"] in LABELS:
            out = AUDIO_DIR / row.filename
            if not out.exists():
                audio = row.audio
                raw = audio["bytes"] if isinstance(audio, dict) else audio.bytes
                out.write_bytes(raw)
            chosen.append({"filename": row.filename, "audio_path": str(out), "actor": int(actor), "label": row["style"]})
    manifest = pd.DataFrame(chosen).sort_values(["actor", "label"]).reset_index(drop=True)
    assert len(manifest) == 96, f"expected 96 rows, got {len(manifest)}"
    assert manifest.groupby("label").size().to_dict() == {x: 24 for x in LABELS}
    manifest.to_csv(MANIFEST_PATH, index=False)
    return manifest


def parse_label(text):
    cleaned = text.lower().strip()
    hits = [label for label in LABELS if re.search(rf"\b{label}\b", cleaned)]
    return hits[0] if len(hits) == 1 else "invalid"


def locate_fusion(model):
    matches = [m for m in model.alm.modules() if hasattr(m, "vq_adaptor")]
    assert len(matches) == 1, f"expected one fusion module, got {len(matches)}"
    return matches[0]


def load_completed():
    completed = set()
    if PRED_PATH.exists():
        for line in PRED_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                completed.add((row["filename"], row["mode"]))
    return completed


def summarize():
    rows = [json.loads(x) for x in PRED_PATH.read_text(encoding="utf-8").splitlines() if x.strip()]
    df = pd.DataFrame(rows)
    summary = {}
    full = df[df["mode"] == "full"].set_index("filename")["prediction"]
    for mode in MODES:
        part = df[df["mode"] == mode]
        valid = part.prediction.isin(LABELS)
        metrics = {
            "n": int(len(part)),
            "accuracy": float(accuracy_score(part.label, part.prediction)),
            "macro_f1": float(f1_score(part.label, part.prediction, labels=LABELS, average="macro", zero_division=0)),
            "coverage": float(valid.mean()),
            "mean_seconds": float(part.seconds.mean()),
            "confusion_matrix": confusion_matrix(part.label, part.prediction, labels=LABELS).tolist(),
        }
        if mode != "full":
            aligned = part.set_index("filename")["prediction"].reindex(full.index)
            metrics["flip_rate_vs_full"] = float((aligned != full).mean())
        summary[mode] = metrics
    SUMMARY_PATH.write_text(json.dumps({"labels": LABELS, "modes": summary}, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([{"mode": k, **{x: y for x, y in v.items() if x != "confusion_matrix"}} for k, v in summary.items()]).to_csv(OUT_DIR / "ravdess_expanded_summary.csv", index=False)
    return summary


def main():
    manifest = prepare_manifest()
    completed = load_completed()
    model = KimiAudio(model_path=str(ROOT / "models/Kimi-Audio-7B-Instruct"), load_detokenizer=False)
    fusion = locate_fusion(model)
    prompt = "Classify the speaker's vocal emotion. Reply with exactly one lowercase label: neutral, happy, sad, or angry."
    for mode in MODES:
        fusion.kfair_branch_mode = mode
        for row in manifest.to_dict("records"):
            key = (row["filename"], mode)
            if key in completed:
                continue
            messages = [
                {"role": "user", "message_type": "text", "content": prompt},
                {"role": "user", "message_type": "audio", "content": row["audio_path"]},
            ]
            torch.cuda.reset_peak_memory_stats()
            started = time.time()
            try:
                _, output = model.generate(
                    messages,
                    output_type="text",
                    max_new_tokens=12,
                    text_temperature=0.0,
                    text_top_k=5,
                )
                error = None
            except Exception as exc:
                output, error = "", repr(exc)
            result = {
                **row,
                "mode": mode,
                "raw_output": output,
                "prediction": parse_label(output),
                "seconds": round(time.time() - started, 4),
                "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2**30, 4),
                "error": error,
            }
            with PRED_PATH.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
            print(json.dumps(result, ensure_ascii=False), flush=True)
    print(json.dumps(summarize(), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
