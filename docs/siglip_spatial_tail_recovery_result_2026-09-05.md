# SigLIP spatial-tail recovery: quality recovered, interaction rejected

The repaired sole DGX development run completed exit 0 from source revision
`498a10d34b9095fa92a7f45f8e0af6031b33344d`. It used 39 fitting classes and
ten disjoint internal-development classes. External labels 49 through 81 were
not accessible. The result is claim-ineligible and does not establish external
benchmark quality or latency.

## Development quality

| Cell | Space | R@1 | mAP@R |
| --- | --- | ---: | ---: |
| depth-18 baseline | self | 753/823 (91.4945%) | 0.3655674314 |
| depth-18 baseline | teacher gallery | 206/823 (25.0304%) | 0.1864482015 |
| tokenwise control | self | 820/823 (99.6355%) | 0.9672243915 |
| tokenwise control | teacher gallery | 573/823 (69.6233%) | 0.6631126913 |
| latent interaction | self | 821/823 (99.7570%) | 0.9403388383 |
| latent interaction | teacher gallery | 598/823 (72.6610%) | 0.6886430377 |
| teacher | self / teacher gallery | 823/823 (100%) | 0.9998447992 |

The latent-interaction arm closed 97.1429% of the baseline-to-teacher R@1 gap
and 90.6183% of the mAP@R gap. It nevertheless failed the preregistered causal
gate because its self-space mAP@R was below the simpler tokenwise control
(0.940339 versus 0.967224). The canonical classification is
`interaction-rejected`; no latency run was started.

This result shows that a small learned tail can recover most of the teacher's
class-disjoint development geometry from depth-18 tokens. It does not show that
cross-token interaction is responsible: the tokenwise control is stronger on
the primary self-retrieval ranking metric, while interaction improves only one
R@1 hit and improves cross-space compatibility. It also does not show external
generalization because the external split remains sealed.

## Optimization and resources

- Tokenwise loss: 1.7532988623 to 0.5113207232 over 4,000 updates.
- Interaction loss: 1.7351205634 to 0.5686948317 over 4,000 updates.
- Whole guarded run: 1,350,926,480,835 ns (22.515 minutes).
- Peak RSS: 35,557,244,928 bytes (33.12 GiB).
- Peak CUDA allocation reported by the wrapper: 33,266 MiB.
- Peak memory PSI full avg10: 0.14; terminal avg10: 0.00.
- Swap changed from 688,600 KiB to 688,720 KiB (+120 KiB).

The first attempt from revision `f226ad5c075a5c4d1e1b1b9d2f397fa64acbaaa0`
terminated safely with exit 125 and `STOP:psi-immediate` before any quality
metric existed. Root cause was whole-cache FP64 materialization during residual
scale calculation. Revision `498a10d` replaced it with fixed row-block FP64
sufficient-statistic accumulation without changing the scientific objective or
weakening the pressure threshold. The repaired run crossed that boundary with
bounded memory. The failed attempt is infrastructure evidence only.

## Exact evidence

- Canonical result:
  `/tmp/sfora-spatial-tail-498a10d34b9095fa92a7f45f8e0af6031b33344d.json`,
  116,352 bytes, exactly one trailing LF, SHA-256
  `5a4ee26eca3218de87e6aafa0464077ebb8a840e79a7c447b5c271da288cfc2d`.
- Sealed artifact:
  `/home/riomus/sfora-spatial-tail-recovery/498a10d34b9095fa92a7f45f8e0af6031b33344d/spatial-tail.safetensors`,
  67,973,944 bytes, SHA-256
  `cf12e5eced83f23327b15919bcfe7f0cd15e01b184c7945bac14c3f80d445fd9`.
- The remote result digest matches the local canonical result. The result
  validator independently recomputed every per-query summary and decision.
- Remote partial output is absent; timeout, Python, and GPU process clearance
  was verified after completion.

## Decision

Do not promote the interaction arm and do not spend the external evaluation
split. The simpler tokenwise arm is the evidence-supported compact candidate,
but it must first pass the unchanged six-cell full-path latency gate. Any later
external evaluation must use the already sealed artifact and a frozen decision
rule; it may not tune on the external labels.
