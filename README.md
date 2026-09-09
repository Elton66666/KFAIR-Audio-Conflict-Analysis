# K-FAIR: Semantic–Prosodic Conflict Analysis and Repair in Kimi-Audio

This repository is the complete research artifact for an ongoing study of how Kimi-Audio-7B-Instruct resolves conflicts between lexical meaning and vocal emotion. It contains the experiment code, data manifests, per-example predictions, statistics, figures, checkpoints, logs, failed attempts, and Chinese progress reports produced from the beginning of the project through the current stage.

The central finding is not that the model “cannot hear prosody.” Factorial interventions show that continuous acoustic features strongly affect internal emotion scores. However, on explicit semantic–prosodic conflicts, the final answer can still be biased toward lexical meaning. Activation patching localizes a causally sensitive late-layer region, and small frozen-backbone repair modules substantially improve acoustic-emotion following.

## Start here

For a human reader:

1. Read [`PROJECT_INDEX.md`](PROJECT_INDEX.md).
2. Read [`reports/KFAIR_中文阶段总报告.md`](reports/KFAIR_中文阶段总报告.md) for the earlier mechanism stage.
3. Read [`reports/KFAIR_本次工作总报告_三项优先工作与实践验证.md`](reports/KFAIR_本次工作总报告_三项优先工作与实践验证.md) for the latest work.
4. Use [`CLAIMS_AND_EVIDENCE.md`](CLAIMS_AND_EVIDENCE.md) to trace every major claim to raw results.

For Claude Code or another coding agent: read [`CLAUDE.md`](CLAUDE.md) first and follow its reading order.

## Headline results

- The held-out-actor baseline SER is 41.15% on the controlled RAVDESS factorial task.
- A 57,344-parameter layer-25 decision-token adapter reaches 90.63% SER.
- A 57,344-parameter layer-22 full-sequence adapter reaches 91.67% SER and 92.71% acoustic following on conflict cells.
- An equal-parameter ordinary LoRA reaches 88.02% SER; it is weaker on clean conflict accuracy but more stable under additive white noise.
- RAVDESS-trained repair transfers directionally to the post-model-release EMIS subset: conflict acoustic following rises from 33.33% to 44.44%, although confidence intervals overlap.
- On the current 32-utterance short-sentence ASR preservation check, all compared methods retain 0 WER.

These numbers are specific to the documented controlled tasks. They must not be presented as general real-world accuracy.

## What is and is not included

Included: all research-relevant source code, patches, manifests, reports, figures, structured results, raw predictions, run logs, and small adapter/LoRA checkpoints.

Not redistributed: Kimi-Audio model weights, Python virtual environments/caches, and original RAVDESS/TESS/EMIS audio. See [`DATA.md`](DATA.md), [`MODEL.md`](MODEL.md), and [`REPRODUCE.md`](REPRODUCE.md).

## Repository status

This is a research snapshot intended for advisor review and paper drafting, not a final camera-ready release. See [`LIMITATIONS.md`](LIMITATIONS.md) before writing strong claims.

