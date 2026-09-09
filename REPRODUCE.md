# Reproduction guide

## Hardware used

- NVIDIA A800 80GB PCIe
- Historical server root: `/root/autodl-tmp/kfair`

## Environment

The original virtual environments are intentionally excluded. Use `environment/requirements.txt`, `environment/pyproject.toml`, and `environment/pip_freeze.txt` to reconstruct dependencies. CUDA/PyTorch/flash-attention compatibility may require platform-specific adjustment.

## General sequence

1. Obtain the official Kimi-Audio code/weights and required tokenizer submodules.
2. Obtain RAVDESS, TESS, and EMIS from their official sources.
3. Recreate the paths described in `DATA.md`, or edit the scripts’ `ROOT` constants.
4. Apply the retained branch-control patches where needed.
5. Follow `TIMELINE.md` and the report sequence in `PROJECT_INDEX.md`.

## Experiment entry points

- Core factorial: `root_scripts/run_kfair_factorial.py`
- Controls and counterfactual ASR: `root_scripts/run_kfair_controls.py`, `run_kfair_counterfactual_asr.py`
- TESS factorial: `root_scripts/run_tess_factorial.py`
- Layer trace: `root_scripts/run_kfair_layer_trace.py`
- Activation patching: `root_scripts/run_activation_patching.py`, `code/run_decision_token_patching.py`
- EMIS time-out: `code/run_emis_temporal_holdout.py`
- Repair prototype/sweep: `code/run_heldout_actor_adapter.py`, `code/run_adapter_sweep.py`
- Practical baselines: `code/run_practical_baselines.py`
- Stage-3 preservation/robustness: `code/run_stage3_*.py`

## Verification

Use the row counts, metrics, and checkpoint summaries in `CLAIMS_AND_EVIDENCE.md` and `provenance/FILE_MANIFEST.csv`. The repository preserves historical logs and failed attempts so exact development history can be inspected.

Some scripts were written as one-off research jobs rather than a unified package. Reproduction should start from the documented environment and paths; refactoring may change behavior and should be kept separate from historical artifacts.

