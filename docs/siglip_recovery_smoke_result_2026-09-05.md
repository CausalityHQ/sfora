# Fixed SigLIP recovery smoke: numerical and budget gates pass

The sole original ten-update-per-arm DGX run completed exit0. This is
disposable training feasibility evidence, not recovered retrieval quality.
No trained checkpoint or evaluation result was retained.

| Arm | Mean seconds/update | Minimum | Maximum |
| --- | --- | --- | --- |
| PA-only | 24.814078492 | 24.538044904 | 26.109755068 |
| PA + teacher relational CE | 32.409055056 | 32.221853642 | 33.088685370 |

All20 updates have finite positive gradient norms and exactly zero observed
descriptor-replay disagreement. Every student parameter received10 optimizer
steps. Both arms start at the same pruned state; their10crop/label hashes match.
The independent teacher state digest is unchanged. Loss values are optimization
diagnostics, not retrieval quality or proof of generalization.

## Budget

The independently recomputed conservative total is18,120.737725seconds,
**5.033538hours**, below the unchanged6GPU-hour cap:

- Already spent:761.489323s speed preflight +589.932864s smoke script.
- Repeated future non-step startup/setup overhead:17.701529s.
- Future updates:198times the sum of both measured maximum step times,
  multiplied by1.25headroom.
- Evaluation allowance1800s; terminal checkpoint allowance300s.

This is a projection, not a guarantee or an ETA for publication. Subsequent
training must enforce the remaining total budget and reset to initial weights;
smoke optimizer states may not be reused.

## Exact evidence

- Result `/home/riomus/siglip-recovery-smoke-4182deddf5be.json`, local `/tmp/`
  copy with the same basename. SHA256
  `0481b835f594cbc9f910c40259a5d40c1958f236f51bbea49104cfdcaffd0344`.
- Original session56404, PGID1782125/child1782126, wrapper592.802436s,
  exit0/no stop/no restart. Subsequent process and GPU clearance verified.
- Log SHA256`47071d0c260ac3d49c391a315fdb68bbddf521053b6b03bff0356947177d02c9`.
- Runner SHA256`4182deddf5be7af0fd538bb0fe197914e18f2f015e59fd2eb238dc64196690e7`.
  Immutable root `/home/riomus/sfora-recovery-smoke-4182deddf5be`.
- Peak CUDA allocated30,923,699,200B, reserved38,807,797,760B;
  peak process RSS11,388,481,536B. Monitor PSI full avg10 peaked0.54, below the
  immediate0.79stop and not sustained for3samples>=0.50. No swap growth.
- Both prior speed and metadata-first optimization-input proofs authenticated
  before training. Only3963optimization image rows decoded; no evaluation band.
- Independent stdlib-only check reproduced every update count, paired hashes,
  teacher flag, raw timing maxima, projection and monitor-result digest binding.

## Verification and next work

153combined new/related tests pass. Scoped Ruff/format and targeted mypy on
five source files pass. This is not repository-wide release assurance.

The final-only198-update engine and exclusive student checkpoint writer are
tested locally; the scientific paired-run CLI and post-sealing evaluator are
still incomplete. Those are the next steps. The target is preserved quality
with the already verified~31%pipeline speed gain; successful compression is
still an engineering baseline, not the overarching novel matched-SOTA claim.
