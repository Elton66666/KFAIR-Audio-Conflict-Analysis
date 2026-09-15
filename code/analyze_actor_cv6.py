import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path("/root/autodl-tmp/kfair")
OUT = ROOT / "outputs"
SUMMARY = OUT / "actor_cv6_run_summary.jsonl"
BASELINE = OUT / "actor_cv6_baseline_predictions.jsonl"
AGGREGATE = OUT / "actor_cv6_aggregate.csv"
FOLDS = OUT / "actor_cv6_fold_seed_metrics.csv"
ACTORS = OUT / "actor_cv6_actor_metrics.csv"
PAIRED = OUT / "actor_cv6_paired_effects.csv"

METRICS = ["ser", "aligned_ser", "conflict_acoustic_follow", "conflict_semantic_follow", "conflict_margin"]


def metric(frame):
    frame = frame.copy()
    frame["correct"] = frame.prediction.eq(frame.acoustic_label)
    frame["semantic"] = frame.prediction.eq(frame.discrete_label)
    conflict = frame[frame.conflict]
    aligned = frame[~frame.conflict]
    return {
        "ser": frame.correct.mean(),
        "aligned_ser": aligned.correct.mean(),
        "conflict_acoustic_follow": conflict.correct.mean(),
        "conflict_semantic_follow": conflict.semantic.mean(),
        "conflict_margin": conflict.acoustic_margin.mean(),
    }


def main():
    records = [json.loads(line) for line in SUMMARY.read_text(encoding="utf-8").splitlines()]
    run_rows = []
    for record in records:
        run_rows.append({
            "fold": record["fold"],
            "method": record["method"],
            "seed": record["seed"],
            "best_epoch": record["best_epoch"],
            "seconds": record["seconds"],
            **record["test"],
        })

    baseline_frame = pd.read_json(BASELINE, lines=True)
    for fold, group in baseline_frame.groupby("fold"):
        run_rows.append({"fold": fold, "method": "baseline", "seed": 0, "best_epoch": 0, "seconds": 0.0, "n": len(group), **metric(group)})
    runs = pd.DataFrame(run_rows).sort_values(["method", "fold", "seed"])
    runs.to_csv(FOLDS, index=False)

    aggregate_rows = []
    for method, group in runs.groupby("method"):
        # First average seeds inside each fold; then treat six actor folds as independent reporting units.
        fold_means = group.groupby("fold")[METRICS].mean()
        row = {"method": method, "n_folds": len(fold_means), "n_runs": len(group)}
        for column in METRICS:
            row[f"{column}_mean"] = fold_means[column].mean()
            row[f"{column}_std"] = fold_means[column].std(ddof=1)
            row[f"{column}_min_fold"] = fold_means[column].min()
            row[f"{column}_max_fold"] = fold_means[column].max()
        aggregate_rows.append(row)
    pd.DataFrame(aggregate_rows).sort_values("method").to_csv(AGGREGATE, index=False)

    # Paired fold-level effects: every method is compared with the baseline on
    # exactly the same four held-out actors. Seeds are averaged inside a fold,
    # leaving six independent actor groups for the uncertainty calculation.
    baseline_by_fold = runs[runs.method.eq("baseline")].set_index("fold")
    paired_rows = []
    for method, group in runs[~runs.method.eq("baseline")].groupby("method"):
        method_by_fold = group.groupby("fold")[METRICS].mean()
        row = {"method": method, "n_folds": len(method_by_fold)}
        for column in METRICS:
            delta = method_by_fold[column] - baseline_by_fold.loc[method_by_fold.index, column]
            try:
                p_value = float(wilcoxon(delta, alternative="two-sided").pvalue)
            except ValueError:
                p_value = 1.0
            row[f"delta_{column}_mean"] = float(delta.mean())
            row[f"delta_{column}_std"] = float(delta.std(ddof=1))
            row[f"delta_{column}_min_fold"] = float(delta.min())
            row[f"delta_{column}_max_fold"] = float(delta.max())
            row[f"delta_{column}_wilcoxon_p"] = p_value
        paired_rows.append(row)
    pd.DataFrame(paired_rows).sort_values("method").to_csv(PAIRED, index=False)

    prediction_frames = [baseline_frame]
    adapted = pd.read_json(OUT / "actor_cv6_test_predictions.jsonl", lines=True)
    prediction_frames.append(adapted)
    actor_rows = []
    for frame in prediction_frames:
        for (method, actor), group in frame.groupby(["method", "actor"]):
            # Adapted rows include three seeds. Report mean across seed-specific actor metrics.
            if method == "baseline":
                values = [metric(group)]
            else:
                values = [metric(seed_group) for _, seed_group in group.groupby("seed")]
            row = {"method": method, "actor": actor, "n_seeds": len(values)}
            for column in METRICS:
                row[column] = float(np.mean([value[column] for value in values]))
            actor_rows.append(row)
    pd.DataFrame(actor_rows).sort_values(["method", "actor"]).to_csv(ACTORS, index=False)

    print(pd.read_csv(AGGREGATE).to_string(index=False))
    print(pd.read_csv(PAIRED).to_string(index=False))
    print(f"WROTE {AGGREGATE} {FOLDS} {ACTORS} {PAIRED}")


if __name__ == "__main__":
    main()
