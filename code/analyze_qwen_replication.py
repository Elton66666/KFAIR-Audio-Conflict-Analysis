import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


ROOT = Path("/root/autodl-tmp/kfair")
OUT = ROOT / "outputs"
BEHAVIOR = OUT / "qwen_behavioral_predictions.jsonl"
ADAPTER = OUT / "qwen_adapter_predictions.jsonl"
COMPARISON = OUT / "qwen_replication_comparison.csv"
EFFECTS = OUT / "qwen_replication_paired_effects.csv"
SUMMARY = OUT / "qwen_replication_analysis.json"


def read_jsonl(path):
    return pd.DataFrame(map(json.loads, path.read_text(encoding="utf-8").splitlines()))


def score(frame):
    correct = frame.prediction.eq(frame.acoustic_label)
    semantic = frame.prediction.eq(frame.semantic_label)
    return {
        "n": int(len(frame)),
        "acoustic_accuracy": float(correct.mean()),
        "semantic_follow": float(semantic.mean()),
        "mean_acoustic_margin": float(frame.acoustic_margin.mean()),
    }


def paired_bootstrap(base, adapted, repetitions=10000):
    joined = base[["filename", "prediction", "acoustic_label", "acoustic_margin", "cluster_id"]].merge(
        adapted[["filename", "prediction", "acoustic_label", "acoustic_margin"]],
        on=["filename", "acoustic_label"],
        suffixes=("_base", "_adapted"),
        validate="one_to_one",
    )
    joined["delta_correct"] = (
        joined.prediction_adapted.eq(joined.acoustic_label).astype(float)
        - joined.prediction_base.eq(joined.acoustic_label).astype(float)
    )
    joined["delta_margin"] = joined.acoustic_margin_adapted - joined.acoustic_margin_base
    clustered = joined.groupby("cluster_id")[["delta_correct", "delta_margin"]].agg(["sum", "count"])
    rng = np.random.default_rng(20260915)
    indices = rng.integers(0, len(clustered), size=(repetitions, len(clustered)))
    output = {}
    for metric in ("delta_correct", "delta_margin"):
        sums = clustered[(metric, "sum")].to_numpy()
        counts = clustered[(metric, "count")].to_numpy()
        samples = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
        output[metric] = {
            "mean": float(joined[metric].mean()),
            "ci_low": float(np.quantile(samples, 0.025)),
            "ci_high": float(np.quantile(samples, 0.975)),
        }
    base_correct = joined.prediction_base.eq(joined.acoustic_label)
    adapted_correct = joined.prediction_adapted.eq(joined.acoustic_label)
    improved = int((~base_correct & adapted_correct).sum())
    harmed = int((base_correct & ~adapted_correct).sum())
    discordant = improved + harmed
    output["mcnemar_exact_p"] = float(binomtest(min(improved, harmed), discordant, 0.5).pvalue) if discordant else 1.0
    output["improved"] = improved
    output["harmed"] = harmed
    return output


def main():
    behavior = read_jsonl(BEHAVIOR).drop_duplicates(["dataset", "filename"], keep="last")
    adapter = read_jsonl(ADAPTER).drop_duplicates(["seed", "split", "filename"], keep="last")
    behavior["actor"] = pd.to_numeric(behavior.get("actor"), errors="coerce")
    subsets = {
        "ravdess_test": behavior[behavior.dataset.eq("RAVDESS") & behavior.actor.ge(21)],
        "emis_all": behavior[behavior.dataset.eq("EMIS")],
        "emis_aligned": behavior[behavior.dataset.eq("EMIS") & ~behavior.conflict],
        "emis_conflict": behavior[behavior.dataset.eq("EMIS") & behavior.conflict],
    }
    rows = []
    effects = []
    seeds = sorted(int(seed) for seed in adapter.seed.unique())
    for subset, base in subsets.items():
        base_score = score(base)
        rows.append({"method": "base", "seed": "base", "subset": subset, **base_score})
        split = "test" if subset == "ravdess_test" else "emis_zero_shot"
        for seed in seeds:
            adapted = adapter[adapter.seed.eq(seed) & adapter.split.eq(split)]
            if subset == "emis_aligned":
                adapted = adapted[~adapted.conflict]
            elif subset == "emis_conflict":
                adapted = adapted[adapted.conflict]
            adapted_score = score(adapted)
            rows.append({"method": "decision_adapter_layer25", "seed": seed, "subset": subset, **adapted_score})
            effect = paired_bootstrap(base, adapted)
            effects.append(
                {
                    "seed": seed,
                    "subset": subset,
                    "delta_accuracy": effect["delta_correct"]["mean"],
                    "delta_accuracy_ci_low": effect["delta_correct"]["ci_low"],
                    "delta_accuracy_ci_high": effect["delta_correct"]["ci_high"],
                    "delta_margin": effect["delta_margin"]["mean"],
                    "delta_margin_ci_low": effect["delta_margin"]["ci_low"],
                    "delta_margin_ci_high": effect["delta_margin"]["ci_high"],
                    "mcnemar_exact_p": effect["mcnemar_exact_p"],
                    "improved": effect["improved"],
                    "harmed": effect["harmed"],
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(COMPARISON, index=False)
    effect_table = pd.DataFrame(effects)
    effect_table.to_csv(EFFECTS, index=False)
    aggregate = []
    for subset, group in table[table.method.ne("base")].groupby("subset"):
        aggregate.append(
            {
                "subset": subset,
                "seeds": len(group),
                "accuracy_mean": float(group.acoustic_accuracy.mean()),
                "accuracy_std": float(group.acoustic_accuracy.std(ddof=1)),
                "margin_mean": float(group.mean_acoustic_margin.mean()),
                "margin_std": float(group.mean_acoustic_margin.std(ddof=1)),
            }
        )
    summary = {
        "model": "Qwen/Qwen2.5-Omni-7B",
        "revision": "ae9e1690543ffd5c0221dc27f79834d0294cba00",
        "baseline": {name: score(frame) for name, frame in subsets.items()},
        "adapter_mean_std": aggregate,
        "paired_effects": effects,
        "interpretation_guardrail": (
            "Qwen has a continuous audio encoder rather than Kimi's separable native "
            "discrete/continuous audio streams. Results support cross-model behavioral "
            "replication and low-rank correction, not exact internal stream equivalence."
        ),
    }
    SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(table.to_string(index=False), flush=True)
    print(effect_table.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
