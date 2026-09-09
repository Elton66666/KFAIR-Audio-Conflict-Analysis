# Instructions for Claude Code

Your task when reading this repository is to reconstruct the full research history accurately and, when requested, help draft a paper without overstating the evidence.

## Required reading order

1. `README.md`
2. `PROJECT_INDEX.md`
3. `TIMELINE.md`
4. `reports/KFAIR_中文阶段总报告.md`
5. `reports/KFAIR_本次工作总报告_三项优先工作与实践验证.md`
6. `CLAIMS_AND_EVIDENCE.md`
7. `LIMITATIONS.md`
8. Relevant experiment code and raw/statistical outputs for any claim you intend to use

## Interpretation rules

- Treat structured result files and per-example predictions as primary experimental evidence. Reports are explanations and summaries.
- Distinguish three questions: average internal stream influence, final behavior under explicit conflict, and post-hoc repair performance. Do not collapse them into one statement.
- The correct combined interpretation is: continuous acoustic features strongly influence internal emotion scores, yet explicit lexical–prosodic conflict can still produce lexical bias at the final decision stage.
- “E/U/P” refer to exposure-risk groups, not guarantees of contamination status. U means not disclosed in the public SER-SFT list; P is post-release/time-out evidence.
- EMIS is synthetic and its external repair confidence intervals overlap. Describe it as directional or promising transfer evidence, not definitive generalization.
- The 0-WER preservation result uses two short fixed RAVDESS sentences. Never generalize it to open-vocabulary ASR.
- Full-sequence adapter is best on clean controlled accuracy; ordinary LoRA is more stable in the current white-noise stress test. Do not claim uniform adapter dominance.
- The surprising baseline improvement at 10 dB is not evidence that noise improves general ability; it may reflect disruption of a competing evidence stream.
- Results come from one primary model family. Second-model replication remains undone.

## Paper-drafting guidance

The safest contribution chain is:

1. Native dual-stream 2×2 factorial intervention with parallel-emotion speech;
2. Base/Instruct and exposure-risk comparisons;
3. late-layer causal localization using activation patching;
4. causally guided parameter-efficient repair;
5. held-out-speaker, time-out, capability-preservation, and noise checks.

Avoid novelty claims such as “first to study emotion mechanisms” unless a fresh literature review supports them. Prefer: “to our knowledge, the first systematic model-internal factorial intervention of Kimi-Audio’s native discrete and continuous audio streams using parallel-emotion speech,” subject to verification.

## Evidence discipline

Before quoting a number, locate it in `CLAIMS_AND_EVIDENCE.md`, open the listed statistical file, and check the corresponding per-example output. Mention sample size and evaluation split. If reports disagree, use the latest raw/statistical output and document the discrepancy.

Files prefixed `failed_` and early smoke tests are provenance, not formal evidence. Run logs establish execution history but are not substitutes for result tables.

