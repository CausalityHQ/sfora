# Ninth blind continuation: EVPC near-miss

Date: 2026-08-05.

The corrected prompt was frozen and pushed before output at
`docs/fable_blind_prompt_pass9_2026-08-05.txt`. Native jobs
`f47754709f5f40e9` (Fable) and `a7c2fbbec01148eb` (direct Opus fallback)
both failed at the consultation-runner layer before producing a receipt or any
provider event. The documented `devbox-ask fable` fallback then ran the exact
same frozen prompt. Fable exhausted its USD 2 budget before completing and the
harness automatically retried the prompt with Claude Opus. Opus returned
`NONE`. The exact answer is preserved at
`docs/fable_blind_output_pass9_2026-08-05.txt`.

This is not a candidate under `docs/search_protocol.md`; no separate frozen
proposal review, diagnostic, preregistration, implementation, or GPU follows.

## Strongest near-miss

Extreme-Value Partition Correction (EVPC) fits a generalized Pareto tail to
per-anchor negative proxy similarities and attempts to scale the proxy
partition from `C` observed negative classes to deployment gallery size `M`.
The proposal correctly kills its own headline mechanism. In its stated loss,

```
Z_M(theta) = (M / C) Z_tail(theta),
```

so `log Z_M = log(M/C) + log Z_tail`. The only `M` dependence is constant in
the network parameters and has zero gradient. The GPD-modified `Z_tail` can
still differ from the ordinary softmax partition, but that residue is no
longer a gallery-size correction.

The proposed return-level repair would restore a nonconstant dependence on
`M`, but it estimates the fitted training-class score tail rather than an
identified unseen-class tail. More decisively for Gate 2, the very recent
WEINCE preprint (*When Softmax Fails at the Top: Extreme Value Corrections for
InfoNCE*, arXiv:2606.00262) already uses anchor-wise online negative-score tail
statistics and a Weibull endpoint-shortfall correction to replace/blend
softmax logits, without added learned parameters. EVPC's surviving tail-loss
mechanism is therefore occupied even though its exact GPD extrapolation formula
is different.

## Local resolution and corrections

The response's universal claim that no all-class training run can estimate
*any* functional of unseen-class geometry is unsupported and is not adopted as
an impossibility result. It is a useful warning about post-fit seen-class
statistics, not a theorem; class-held-out meta-estimation and assumptions about
exchangeable classes are counterexamples to the categorical wording.

Its uncertainty arithmetic is also only heuristic. PFML's reported dispersion
and an unpaired hypothetical new method do not justify declaring `3 SE` of two
five-run means to be the protocol's universal crossing requirement. The
correct operational requirement remains prospective matched controls,
independent checkpoint selection/final metrics, and replication.

Finally, the answer's suggestion that PFML might be CVPR 2024 is false. The
primary CVF proceedings record is CVPR **2025**, pages 25549--25559. The
canonical frontier file was correct.

Verdict: **NONE / DEAD near-miss at algebra and Gate 2.** No GPU follows.
