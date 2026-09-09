# Claims and evidence map

Use this file to verify potential paper statements. Paths below refer to the reorganized repository.

| Potential claim | Main evidence | Raw evidence | Important qualification |
|---|---|---|---|
| Continuous acoustic input has a larger average effect than the discrete stream on the controlled emotion readout | `results/statistics/kfair_factorial_effects.csv`, `tess_factorial_effects.csv` | `results/raw_predictions/kfair_factorial_predictions.jsonl`, `tess_factorial_predictions.jsonl` | “Discrete” and “continuous” are semantic-dominant/acoustic-rich, not pure factors |
| The strong task-readable late-layer effect is associated with instruction tuning | `results/statistics/kfair_base_factorial_statistics.csv`, layer-trace statistics | Base/Instruct prediction and layer-trace JSONL files | Fixed readout may underrepresent Base capabilities |
| Late layers causally influence conflict decisions | `results/statistics/kfair_activation_patching_statistics.csv`, `kfair_decision_token_patching_statistics.csv` | corresponding patching JSONL files | Region is approximately layers 22–27; do not reduce it to only 24–27 |
| Time-out EMIS data reproduces lexical bias under conflict | `results/statistics/emis_temporal_holdout_statistics.csv` | `results/raw_predictions/emis_temporal_holdout_predictions.jsonl` | EMIS is synthetic; some bootstrap intervals cross zero |
| A small frozen-backbone repair improves held-out-actor performance | `results/statistics/adapter_sweep_summary.jsonl`, `practical_rounds_comparison.csv` | `results/raw_predictions/adapter_sweep_test_predictions.jsonl` | Controlled four-class factorial task, not universal SER |
| Pair and preserve objectives matter | layer-22 no-pair/no-preserve entries in `adapter_sweep_summary.jsonl` | sweep predictions and checkpoints | Limited seeds for ablation variants |
| Repair transfers directionally from RAVDESS to EMIS | `results/statistics/adapter_emis_zeroshot_statistics.csv` | `results/raw_predictions/adapter_emis_zeroshot_predictions.jsonl` | 33.33%→44.44%; confidence intervals overlap |
| Simple prompt/calibration/gating cannot match learned repair | `results/statistics/practical_baselines_summary.json` | experiment code/log | Ordinary LoRA is a strong baseline and must be reported |
| Decision adapter adds negligible observed forward overhead | `results/statistics/efficiency_benchmark.csv` | benchmark code/log | Small microbenchmark; excludes preprocessing |
| Current short-sentence ASR is preserved | `results/statistics/stage3_asr_preservation_summary.csv` | `results/raw_predictions/stage3_asr_preservation_predictions.jsonl` | Only 32 utterances and two fixed sentences |
| Repair does not degrade the sampled clean TESS set | `results/statistics/stage3_tess_clean_summary.csv`, paired statistics | TESS stage-3 predictions | Only two speakers; neutral performance is near zero |
| Repairs remain useful under white noise | `results/statistics/stage3_noise_robustness_summary.csv`, paired statistics | noise predictions | Ordinary LoRA is more stable than adapters in this stress test |

## Headline controlled-test values

| Method | SER | Aligned SER | Conflict acoustic following |
|---|---:|---:|---:|
| Baseline | 41.15% | 43.75% | 38.54% |
| Layer-25 decision adapter | 90.63% | 93.75% | 87.50% |
| Layer-22 full-sequence adapter | 91.67% | 90.63% | 92.71% |
| Equal-parameter ordinary LoRA | 88.02% | 90.63% | 85.42% |

Always verify exact values against the listed machine-readable files before drafting final tables.

