# Experiment timeline

This is a conceptual timeline; file modification times and logs provide finer-grained provenance.

## Phase A — Interface validation and early ablations

- Established Kimi-Audio inference on the AutoDL A800 environment.
- Tested full, no-continuous, no-discrete, and time-rolled-continuous conditions.
- Evaluated emotion classification and ASR; identified free-generation label-coverage problems.
- Moved to fixed candidate-token likelihoods for cleaner four-class comparisons.

## Phase B — K-FAIR factorial intervention

- Constructed real parallel-emotion pairs with the same speaker and text.
- Ran `D_A/C_A`, `D_A/C_B`, `D_B/C_A`, `D_B/C_B` cells.
- Estimated continuous-stream, discrete-stream, and interaction effects.
- Added counterfactual ASR and distribution-shift controls.

## Phase C — Exposure and post-training comparisons

- Repeated the factorial design on RAVDESS/E and TESS/U.
- Compared Kimi-Audio Base and Instruct.
- Traced layerwise readout and observed late-layer amplification in Instruct.

## Phase D — Three priority tasks

- Used activation patching and decision-token patching to test causal influence in late layers.
- Built a balanced 192-example EMIS/P time-out subset released after the original Kimi-Audio weights.
- Built an initial held-out-actor layer-22 residual-adapter repair prototype.

## Phase E — Practical repair rounds 1 and 2

- Compared layers 12/18/22/25/27 and an early layer-5 control.
- Compared decision-token and full-sequence repair.
- Added three seeds and pair/preserve-loss ablations.
- Tested zero-shot RAVDESS-to-EMIS transfer.
- Compared calibration, acoustic gate, prompt-only, equal-parameter ordinary LoRA, and residual adapters.
- Measured forward latency and GPU memory.

## Phase F — Practical repair round 3

- Expanded held-out short-sentence ASR preservation to all four emotions.
- Evaluated clean TESS transfer on 200 deterministically sampled utterances.
- Tested clean, 20 dB, and 10 dB additive white-noise conditions.
- Added paired exact comparisons.

## Current state

The single-model research chain is complete enough for advisor review and a paper draft. Stronger publication evidence would include second-model replication, natural conflict speech, broader speakers, open-vocabulary ASR, and realistic environmental distortions.

