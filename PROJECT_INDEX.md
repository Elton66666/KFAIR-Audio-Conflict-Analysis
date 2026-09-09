# Project index

## Repository sections

- `code/`: official Kimi-Audio snapshot plus project experiment/analysis scripts used on the server.
- `root_scripts/`: early scripts and patches originally stored at the server project root.
- `archive/local_work/`: supplementary local scripts, subset-building metadata, and early analysis materials.
- `reports/`: all Markdown reports and meeting notes, preserved with original filenames.
- `results/raw_predictions/`: per-example JSONL outputs.
- `results/statistics/`: CSV/JSON summaries and derived comparison tables.
- `results/figures/`: SVG figures.
- `results/logs/`: run logs and failed interface attempts.
- `checkpoints/`: all small residual-adapter and LoRA checkpoints.
- `manifests/`: dataset and factorial-pair manifests; no original audio.
- `environment/`: dependency specifications and captured environment metadata.
- `provenance/`: upstream revision, original paths, file manifest, and integrity hashes.

## Recommended report sequence

1. `reports/stage1_report.md`
2. `reports/stage2_meeting_report.md` and `reports/final_meeting_report.md`
3. `reports/kfair_factorial_report.md`
4. `reports/kfair_eu_comparison_report.md`
5. `reports/kfair_base_instruct_comparison.md` and `reports/kfair_base_tess_report.md`
6. `reports/kfair_layer_trace_report.md`
7. `reports/KFAIR_中文阶段总报告.md`
8. `reports/三项优先工作结果_2026-09-03.md`
9. `reports/实践导向第一二轮结果_2026-09-04.md`
10. `reports/实践导向第三轮结果_2026-09-04.md`
11. `reports/KFAIR_本次工作总报告_三项优先工作与实践验证.md`

The later total reports supersede status statements in earlier reports. Earlier reports remain important for reconstructing decisions and failed assumptions.

## Result families

- Stage 1/2 branch ablation: `ravdess_branch_*`, `ravdess_choice_*`, `ravdess_*emotion*`, `ravdess_*asr*`
- Core RAVDESS factorial: `kfair_factorial_*`, `kfair_pair_manifest.csv`
- Controls and counterfactual ASR: `kfair_controls*`, `kfair_counterfactual_asr*`
- TESS/U factorial: `tess_factorial_*`, `tess_pair_manifest.csv`
- Base/Instruct: `kfair_base_*`, `tess_base_*`
- Layer tracing: `kfair_layer_trace_*`, `kfair_base_layer_trace_*`
- Causal patching: `kfair_activation_patching*`, `kfair_decision_token_patching*`
- EMIS/P: `emis_temporal_holdout*`, `adapter_emis_zeroshot*`
- Repair prototype: `heldout_actor_adapter*`
- Repair sweep and ablations: `adapter_sweep*`
- Practical baselines and efficiency: `practical_baselines*`, `efficiency_benchmark*`
- Capability preservation and robustness: `stage3_*`

