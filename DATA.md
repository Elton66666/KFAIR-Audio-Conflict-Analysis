# Data inventory and redistribution policy

Original audio is intentionally not redistributed in this public repository. Dataset licenses/terms should be checked by each user and the data acquired from official sources.

## RAVDESS

Used for the main controlled factorial construction, Base/Instruct comparisons, causal patching, held-out-actor repair, ASR preservation, and white-noise stress tests. Actor IDs define the repair split: 1–16 train, 17–20 validation, and 21–24 test.

The repository includes all generated manifests, filenames, labels, pair/cell definitions, and predictions required to audit sample use.

## TESS

Used as the U exposure-risk comparison and for a 200-utterance clean transfer check. The stage-3 subset is selected deterministically by filename hash with 25 examples per speaker×emotion cell for the four shared labels.

## EMIS

Used as a P/time-out dataset because its release postdates the initial Kimi-Audio weights. The final external evaluation uses all 1,248 official samples: 936 semantic–acoustic conflicts and 312 aligned examples; COSY, F5TTS, and STYLE contribute 416 each. Four semantic and four acoustic labels each occur 312 times, spanning 26 text IDs and 10 voice IDs. The earlier balanced audit subset contains 192 examples (144 conflict, 48 aligned) and is retained for provenance. EMIS is synthetic, not natural recorded conflict speech. Audio is not redistributed; official source and checksums are recorded in `provenance/emis_full1248_provenance.json`.

## Labels

The common four-class task uses `neutral`, `happy`, `sad`, and `angry`, mapped to candidate output tokens A/B/C/D. Consult manifests and experiment scripts for exact prompt wording and pair generation.

## Expected server paths

Historical scripts expect `/root/autodl-tmp/kfair/data/...`. For another machine, either recreate this layout or parameterize the `ROOT` constants. Absolute paths are preserved for exact provenance and should not be interpreted as portable defaults.
