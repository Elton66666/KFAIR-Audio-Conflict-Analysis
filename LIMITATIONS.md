# Limitations and claim boundaries

1. **One primary model family.** The complete intervention-and-repair chain is currently demonstrated on Kimi-Audio. Second-model replication remains undone.
2. **Controlled conflict construction.** Much of the strongest evidence comes from factorial swapping or synthetic semantic–prosodic conflict, not natural conversational sarcasm.
3. **Dataset exposure categories are imperfect.** E/U/P reduce but do not eliminate contamination concerns. U is based on public disclosure, not proof of absence from pretraining.
4. **EMIS is synthetic and modest in size.** The zero-shot repair gain is directional; confidence intervals overlap.
5. **Small speaker coverage.** RAVDESS held-out tests use actors 21–24. The sampled TESS evaluation contains only two speakers.
6. **ASR preservation is narrow.** The current 0-WER result covers two short fixed English sentences and cannot establish open-vocabulary preservation.
7. **Noise test is limited.** Only additive white Gaussian noise at two SNRs was tested. Real noise, reverberation, devices, codecs, and far-field speech remain untested.
8. **Validation ranking is noisy.** Layer rankings on the four-actor validation split do not perfectly match test rankings; broader nested or cross-validation would improve selection reliability.
9. **No universal method dominance.** Full-sequence adapter is best on the clean controlled test, while ordinary LoRA is more stable under the current white-noise stress test.
10. **Mechanistic labels are shorthand.** The discrete stream is semantic-dominant and the continuous stream acoustic-rich; neither is a pure semantic/acoustic variable.

These limitations should appear explicitly in any draft paper rather than being hidden.

