# Claims and evidence map

Use this file to verify potential paper statements. Paths below refer to the reorganized repository.

| Potential claim | Main evidence | Raw evidence | Important qualification |
|---|---|---|---|
| Continuous acoustic input has a larger average effect than the discrete stream on the controlled emotion readout | `results/statistics/kfair_factorial_effects.csv`, `tess_factorial_effects.csv` | `results/raw_predictions/kfair_factorial_predictions.jsonl`, `tess_factorial_predictions.jsonl` | “Discrete” and “continuous” are semantic-dominant/acoustic-rich, not pure factors |
| The strong task-readable late-layer effect is associated with instruction tuning | `results/statistics/kfair_base_factorial_statistics.csv`, layer-trace statistics | Base/Instruct prediction and layer-trace JSONL files | Fixed readout may underrepresent Base capabilities |
| Late layers causally influence conflict decisions | `results/statistics/kfair_activation_patching_statistics.csv`, `kfair_decision_token_patching_statistics.csv` | corresponding patching JSONL files | Region is approximately layers 22–27; do not reduce it to only 24–27 |
| Full time-out EMIS data confirms that aligned continuous features materially support acoustic emotion decisions | `results/statistics/emis_full1248_summary.csv`, `emis_full1248_analysis.json` | `results/raw_predictions/emis_full1248_predictions.jsonl` | All 1,248 official samples; EMIS remains synthetic |
| A small frozen-backbone repair improves held-out-actor performance | `results/statistics/adapter_sweep_summary.jsonl`, `practical_rounds_comparison.csv` | `results/raw_predictions/adapter_sweep_test_predictions.jsonl` | Controlled four-class factorial task, not universal SER |
| Pair and preserve objectives matter | layer-22 no-pair/no-preserve entries in `adapter_sweep_summary.jsonl` | sweep predictions and checkpoints | Limited seeds for ablation variants |
| RAVDESS-trained repairs transfer modestly but consistently to full EMIS | `results/statistics/emis_full1248_paired_effects.csv`, `emis_full1248_analysis.json` | `results/raw_predictions/emis_full1248_predictions.jsonl` | Conflict acoustic following: 43.59% baseline, 49.15%±1.34% decision adapter; all nine repair-seed paired tests p<0.05; not a claim that cross-domain bias is solved |
| Simple prompt/calibration/gating cannot match learned repair | `results/statistics/practical_baselines_summary.json` | experiment code/log | Ordinary LoRA is a strong baseline and must be reported |
| Decision adapter adds negligible observed forward overhead | `results/statistics/efficiency_benchmark.csv` | benchmark code/log | Small microbenchmark; excludes preprocessing |
| Current short-sentence ASR is preserved | `results/statistics/stage3_asr_preservation_summary.csv` | `results/raw_predictions/stage3_asr_preservation_predictions.jsonl` | Only 32 utterances and two fixed sentences |
| Repair does not degrade the sampled clean TESS set | `results/statistics/stage3_tess_clean_summary.csv`, paired statistics | TESS stage-3 predictions | Only two speakers; neutral performance is near zero |
| Repairs remain useful under white noise | `results/statistics/stage3_noise_robustness_summary.csv`, paired statistics | noise predictions | Ordinary LoRA is more stable than adapters in this stress test |
| The frozen-backbone repair improvement persists across all 24 actors | `results/statistics/actor_cv6_aggregate.csv`, `actor_cv6_paired_effects.csv` | `results/raw_predictions/actor_cv6_test_predictions.jsonl`, 54 checkpoints | Report six-fold mean ± SD; the old 91.67% is a fixed-split peak, not the overall mean |
| Semantic–acoustic conflict bias also occurs in Qwen2.5-Omni-7B | `results/statistics/qwen_behavioral_summary.json`, `qwen_behavioral_statistics.csv` | `results/raw_predictions/qwen_behavioral_predictions.jsonl` | Qwen has a continuous audio encoder, so this is behavioral replication, not Kimi-style native stream swapping |
| A 57,344-parameter layer-25 adapter improves Qwen held-out-actor accuracy | `results/statistics/qwen_replication_analysis.json`, `qwen_replication_comparison.csv` | `results/raw_predictions/qwen_adapter_predictions.jsonl`, Qwen checkpoints | RAVDESS improves strongly; zero-shot EMIS conflict acoustic following remains very low |

## Earlier fixed-split controlled-test values

| Method | SER | Aligned SER | Conflict acoustic following |
|---|---:|---:|---:|
| Baseline | 41.15% | 43.75% | 38.54% |
| Layer-25 decision adapter | 90.63% | 93.75% | 87.50% |
| Layer-22 full-sequence adapter | 91.67% | 90.63% | 92.71% |
| Equal-parameter ordinary LoRA | 88.02% | 90.63% | 85.42% |

Always verify exact values against the listed machine-readable files before drafting final tables.

These values are retained for historical comparability, but the six-fold table below should be the primary actor-generalization result.

## Six-fold held-out-actor values (latest primary result)

| Method | SER (mean ± SD across six actor folds) | Min fold | Max fold |
|---|---:|---:|---:|
| Baseline | 43.66% ± 9.21% | 29.17% | 54.69% |
| Layer-25 decision adapter | 75.52% ± 11.03% | 57.47% | 91.32% |
| Layer-22 full-sequence adapter | 76.13% ± 6.25% | 67.19% | 86.28% |
| Equal-parameter ordinary LoRA | 77.95% ± 10.32% | 63.19% | 93.58% |

All three trainable methods use 57,344 parameters. Their paired six-fold SER gains over baseline have two-sided Wilcoxon `p=0.03125`.

## Qwen2.5-Omni minimal replication

- RAVDESS held-out actors 21–24: 65.63% baseline; layer-25 adapter 84.38% ± 8.27% across three seeds.
- EMIS aligned accuracy: 100% baseline.
- EMIS conflict: 1.39% acoustic following and 95.14% semantic following at baseline.
- After RAVDESS-only adapter training, EMIS conflict acoustic following is 4.40% ± 1.45%; this is only a small cross-domain improvement.
