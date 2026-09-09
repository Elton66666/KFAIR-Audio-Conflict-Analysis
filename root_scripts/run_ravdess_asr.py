import json
import re
import time
from pathlib import Path

import pandas as pd
import torch

from kimia_infer.api.kimia import KimiAudio

ROOT = Path("/root/autodl-tmp/kfair")
OUT = ROOT / "outputs"
MANIFEST = OUT / "ravdess_expanded_manifest.csv"
PRED = OUT / "ravdess_asr_predictions.jsonl"
SUMMARY = OUT / "ravdess_asr_summary.csv"
MODES = ["full", "no_continuous", "no_discrete", "rolled_continuous"]
REFERENCE = "kids are talking by the door"


def words(text):
    return re.findall(r"[a-z]+", text.lower())


def edit_distance(ref, hyp):
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def completed_keys():
    if not PRED.exists():
        return set()
    return {(x["filename"], x["mode"]) for x in map(json.loads, PRED.read_text(encoding="utf-8").splitlines())}


def main():
    manifest = pd.read_csv(MANIFEST)
    assert len(manifest) == 96
    done = completed_keys()
    model = KimiAudio(model_path=str(ROOT / "models/Kimi-Audio-7B-Instruct"), load_detokenizer=False)
    matches = [m for m in model.alm.modules() if hasattr(m, "vq_adaptor")]
    assert len(matches) == 1
    fusion = matches[0]
    prompt = "Transcribe the spoken words exactly. Reply with only the transcript."
    for mode in MODES:
        fusion.kfair_branch_mode = mode
        for row in manifest.to_dict("records"):
            if (row["filename"], mode) in done:
                continue
            messages = [
                {"role": "user", "message_type": "text", "content": prompt},
                {"role": "user", "message_type": "audio", "content": row["audio_path"]},
            ]
            started = time.time()
            try:
                _, text = model.generate(messages, output_type="text", max_new_tokens=48, text_temperature=0.0, text_top_k=5)
                err = None
            except Exception as exc:
                text, err = "", repr(exc)
            ref, hyp = words(REFERENCE), words(text)
            edits = edit_distance(ref, hyp)
            result = {**row, "mode": mode, "reference": REFERENCE, "transcript": text, "edits": edits,
                      "ref_words": len(ref), "wer": edits / len(ref), "exact_match": hyp == ref,
                      "seconds": round(time.time() - started, 4), "error": err}
            with PRED.open("a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
            print(json.dumps(result, ensure_ascii=False), flush=True)
    df = pd.DataFrame(json.loads(x) for x in PRED.read_text(encoding="utf-8").splitlines())
    rows = []
    for mode in MODES:
        p = df[df["mode"] == mode]
        rows.append({"mode": mode, "n": len(p), "wer_micro": p.edits.sum() / p.ref_words.sum(),
                     "wer_mean": p.wer.mean(), "exact_match": p.exact_match.mean(),
                     "mean_seconds": p.seconds.mean(), "errors": p.error.notna().sum()})
    pd.DataFrame(rows).to_csv(SUMMARY, index=False)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
