import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/root/autodl-tmp/kfair")
OUT = ROOT / "outputs"
raw = [json.loads(line) for line in (OUT / "kfair_decision_token_patching.jsonl").read_text(encoding="utf-8").splitlines()]
df = pd.DataFrame(raw).drop_duplicates(["pair_id", "target_cell", "condition", "layer"], keep="last")
assert len(df) == 2880, len(df)
df["acoustic_follow"] = df.prediction.eq(df.acoustic_label)
df["discrete_follow"] = df.prediction.eq(df.discrete_label)
df["other"] = ~df.acoustic_follow & ~df.discrete_follow
actors = sorted(df.actor.unique())
rng = np.random.default_rng(20260903)


def actor_bootstrap(group, column, repetitions=10000):
    values = group.groupby("actor")[column].mean().reindex(actors).to_numpy()
    indices = rng.integers(0, len(values), size=(repetitions, len(values)))
    samples = values[indices].mean(axis=1)
    return float(values.mean()), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


rows = []
for (condition, layer), group in df.groupby(["condition", "layer"], dropna=False):
    for metric in ("acoustic_margin", "rescue", "acoustic_follow", "discrete_follow", "other"):
        mean, low, high = actor_bootstrap(group, metric)
        rows.append({
            "condition": condition,
            "layer": layer,
            "metric": metric,
            "mean": mean,
            "ci_low": low,
            "ci_high": high,
            "n": len(group),
        })
stats = pd.DataFrame(rows)
stats.to_csv(OUT / "kfair_decision_token_patching_statistics.csv", index=False)
print(stats[stats.metric.isin(["rescue", "acoustic_follow", "discrete_follow"])].to_string(index=False))
