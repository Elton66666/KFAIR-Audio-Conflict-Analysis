# Model and upstream provenance

Primary model: `Kimi-Audio-7B-Instruct`, obtained from the official MoonshotAI release. Model weights are not included.

Official repository: <https://github.com/MoonshotAI/Kimi-Audio>  
Historical upstream code commit used on the server: `349251e1d8f4f98d58fda59246381faecd7392e0`

The archived source includes local changes needed to expose/control native branches and all project experiment scripts. Patch files are retained separately under `patches/` and `root_scripts/` for easier auditing.

The study also used the Kimi-Audio Base checkpoint for specific Base/Instruct comparisons. Consult the relevant scripts and reports for exact model paths.

## Small learned artifacts included

- layer-25 decision-token residual adapter;
- layer-22 full-sequence residual adapter;
- equal-parameter layer-22 ordinary LoRA;
- layer/seed/ablation sweep checkpoints.

These `.pt` files contain only small learned modules and summaries, not the 7B backbone.

