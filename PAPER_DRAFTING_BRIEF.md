# Paper drafting brief

## Working problem statement

Audio-language models receive both lexical/semantic and paralinguistic/acoustic evidence. When the spoken words and vocal emotion disagree, a model may extract prosodic evidence internally yet fail to use it consistently at the final decision stage. This project studies that failure in Kimi-Audio’s native dual-stream interface and tests whether causal localization can guide a small repair.

## Suggested paper arc

1. **Problem:** semantic–prosodic conflict exposes unreliable evidence arbitration in audio LLMs.
2. **Methodological contribution:** K-FAIR constructs a model-internal 2×2 factorial intervention over Kimi-Audio’s native discrete semantic-dominant and continuous acoustic-rich streams using parallel-emotion speech.
3. **Mechanistic evidence:** Base/Instruct and layerwise comparisons locate late-stage amplification; activation patching establishes causal sensitivity around layers 22–27.
4. **Practical contribution:** frozen-backbone rank-8 residual repair at a localized layer uses only 57,344 trainable parameters.
5. **Evaluation:** held-out actors, layer controls, multiple seeds, loss ablations, prompt/calibration/gate/equal-parameter LoRA baselines, time-out EMIS transfer, ASR preservation, TESS transfer, and white-noise stress tests.
6. **Balanced conclusion:** localized residual repair is strongest on the clean controlled task and nearly cost-free at the decision token, while ordinary LoRA is more stable under the tested white noise.

## Candidate contributions, pending literature verification

- A controlled factorial framework for separating native dual-stream influence without relying only on branch deletion.
- Evidence that instruction tuning converts or amplifies acoustic emotion information into a strong late-layer task readout.
- Causal localization of conflict arbitration through targeted activation patching.
- A causally guided, parameter-efficient repair with held-out-speaker and external checks.

## Results that should appear in the main paper

- Core RAVDESS and TESS factorial effects with actor/speaker-aware uncertainty.
- Base/Instruct layer comparison.
- Decision-token activation-patching curve around layers 22–27.
- Held-out-actor table containing baseline, prompt, calibration, gate, equal-parameter LoRA, decision adapter, and full-sequence adapter.
- Pair/preserve-loss ablations and three-seed stability.
- EMIS time-out result with confidence intervals and explicit synthetic-data caveat.

## Results better suited to appendix/supplement

- Early branch-zeroing experiments and label-generation failures.
- Full per-emotion tables.
- Random/noise/roll controls.
- Detailed latency/memory microbenchmark.
- Full ASR transcripts and noise-condition paired tables.
- Complete run logs and failed interface attempts.

## Do not write

- Do not claim the model generally ignores acoustics.
- Do not claim discrete tokens are pure semantics or continuous features are pure acoustics.
- Do not claim 91.67% is general real-world SER.
- Do not claim definitive EMIS generalization.
- Do not claim general ASR preservation from two fixed sentences.
- Do not claim adapter superiority under every corruption.
- Do not claim cross-model generality before second-model replication.

