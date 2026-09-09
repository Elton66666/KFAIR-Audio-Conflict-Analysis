import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
PREFIX = os.environ.get("KFAIR_PREFIX", "tess")
pred = pd.DataFrame(json.loads(x) for x in (OUT / f"{PREFIX}_factorial_predictions.jsonl").read_text(encoding="utf-8").splitlines())
eff = pd.read_csv(OUT / f"{PREFIX}_factorial_effects.csv")
rng = np.random.default_rng(20260826)
words = sorted(eff.word.unique())


def cluster_boot(df, col, n=10000):
    vals = df.groupby("word")[col].mean().reindex(words).to_numpy()
    ix = rng.integers(0, len(vals), size=(n, len(vals)))
    samples = vals[ix].mean(axis=1)
    return float(vals.mean()), float(np.quantile(samples, .025)), float(np.quantile(samples, .975))


rows = []
for col in ["ME_C", "ME_D", "interaction"]:
    mean, lo, hi = cluster_boot(eff, col)
    rows.append({"metric": col, "mean": mean, "ci_low": lo, "ci_high": hi})

cross = pred[pred.cell.isin(["AB", "BA"])].copy()
cross["acoustic_follow"] = np.where(cross.cell.eq("AB"), cross.prediction.eq(cross.label_b), cross.prediction.eq(cross.label_a))
cross["discrete_follow"] = np.where(cross.cell.eq("AB"), cross.prediction.eq(cross.label_a), cross.prediction.eq(cross.label_b))
cross["other"] = ~cross.acoustic_follow & ~cross.discrete_follow
for col in ["acoustic_follow", "discrete_follow", "other"]:
    mean, lo, hi = cluster_boot(cross, col)
    rows.append({"metric": col, "mean": mean, "ci_low": lo, "ci_high": hi})

matched = pred[pred.cell.isin(["AA", "BB"])].copy()
matched["correct"] = np.where(matched.cell.eq("AA"), matched.prediction.eq(matched.label_a), matched.prediction.eq(matched.label_b))
mean, lo, hi = cluster_boot(matched, "correct")
rows.append({"metric": "matched_accuracy", "mean": mean, "ci_low": lo, "ci_high": hi})
stats = pd.DataFrame(rows)
stats.to_csv(OUT / f"{PREFIX}_factorial_statistics.csv", index=False)

pair_rows = []
for (a, b), g in eff.groupby(["label_a", "label_b"]):
    c = cross[(cross.label_a == a) & (cross.label_b == b)]
    pair_rows.append({
        "emotion_pair": f"{a}-{b}", "n_pairs": len(g),
        "ME_C": g.ME_C.mean(), "ME_D": g.ME_D.mean(), "interaction": g.interaction.mean(),
        "acoustic_follow": c.acoustic_follow.mean(), "discrete_follow": c.discrete_follow.mean(),
        "other": c.other.mean(),
    })
pair_stats = pd.DataFrame(pair_rows)
pair_stats.to_csv(OUT / f"{PREFIX}_by_emotion_pair.csv", index=False)
print(stats.to_string(index=False))
print(pair_stats.to_string(index=False))
