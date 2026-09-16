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
12. `reports/投稿前高优先级补实验结果_2026-09-16.md` (six-fold actor CV and initial Qwen2.5-Omni replication)
13. `reports/LOW_PRIORITY_THREE_EXPERIMENTS_ZH.md` (latest parameter-matched Qwen control, realistic perturbations, and expanded ASR preservation)
14. `reports/EMIS_FULL1248_ZH.md` (full 1,248-sample time-out evaluation and external repair transfer)
15. `reports/POST_REVIEW_COMPLETE_SUPPLEMENT_ZH.md` (single exhaustive post-review master report; start here when updating the paper, with the three detailed source reports preserved in full)

For paper revision after the external review, start with report 15. The later total reports supersede status statements in earlier reports. Earlier reports remain important for reconstructing decisions and failed assumptions.

## Result families

- Stage 1/2 branch ablation: `ravdess_branch_*`, `ravdess_choice_*`, `ravdess_*emotion*`, `ravdess_*asr*`
- Core RAVDESS factorial: `kfair_factorial_*`, `kfair_pair_manifest.csv`
- Controls and counterfactual ASR: `kfair_controls*`, `kfair_counterfactual_asr*`
- TESS/U factorial: `tess_factorial_*`, `tess_pair_manifest.csv`
- Base/Instruct: `kfair_base_*`, `tess_base_*`
- Layer tracing: `kfair_layer_trace_*`, `kfair_base_layer_trace_*`
- Causal patching: `kfair_activation_patching*`, `kfair_decision_token_patching*`
- EMIS/P: legacy balanced-subset results `emis_temporal_holdout*`, `adapter_emis_zeroshot*`; final full-corpus results `emis_full1248_*`
- Repair prototype: `heldout_actor_adapter*`
- Repair sweep and ablations: `adapter_sweep*`
- Practical baselines and efficiency: `practical_baselines*`, `efficiency_benchmark*`
- Capability preservation and robustness: `stage3_*`
- Six-fold held-out-actor CV: `actor_cv6_*`
- Qwen2.5-Omni cross-model replication: `qwen_behavioral_*`, `qwen_adapter_*`, `qwen_replication_*`
- Parameter-matched Qwen control: `qwen_parameter_matched_*`
- Realistic SLR28/G.711 robustness: `realistic_robustness_*`
- Expanded 384-utterance ASR preservation: `expanded_asr_preservation_*`
