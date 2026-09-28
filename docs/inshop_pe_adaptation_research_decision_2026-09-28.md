# Reject the displacement-to-LR proposal

Fable consultation `575a2d205fe6486d` completed normally in547 seconds,
exit0, within its $6/1,800-second caps (not measured spend). Its proposed
mechanism was a function-space step mismatch: measure original-to-final raw
angular displacement on the held matrices, then multiply PE's native AdamW
learning rate by a ratio derived from Large's displacement, clipped to1.5–3.
It proposed closing PE on capacity grounds if a displacement ratio failed.
The [complete research answer](evidence/compact_metric/sop-siglip2-substrate-v1/pe-update-geometry-v1/research-result.json)
is preserved as a recommendation, not measured causal evidence.

Decision: **STOP this gate before corpus scoring or new training.** Keep the
100-update quality STOP. This does not prove a capacity ceiling or close every
possible PE adaptation mechanism. No supported distinct intervention survives
this recommendation; optimization versus capacity remains unresolved.

The cheapest [runnable premise check](../scripts/check_inshop_pe_displacement_premise.py)
establishes two concrete problems using stdlib arithmetic:

- A common signed90-degree coordinate permutation moves every sample by90
  degrees while preserving all tested float cosine and signed-int8 code dot
  products and norms. Thus raw displacement need not measure any change in
  retrieval. This signed permutation preserves packed geometry; arbitrary
  rotations generally preserve float cosine but need not preserve quantization.
- Bias-corrected Adam with default betas and scalar gradients0.9,1, both within
  clip1, takes a second step1.00135778 times LR. Continuing with unit gradients
  through update100 gives total100.19631215 times LR. Both the strict per-step
  LR bound and the proposed100×LR×sqrt(n) bound fail for this n=1 example.
  This arithmetic counterexample is not a new optimizer or training run.

Other inference failures remain even if the raw statistic were calculated:
small displacement can reflect cancellation or unproductive directions,
and large displacement can reflect changes common to every image. Neither
outcome separates insufficient LR from capacity. Matching one median across
different pretrained towers does not establish a quality-preserving control.
Native pooling/projection contributes to the statistic too. AdamW LR changes
also change the amount of decoupled weight decay, so the coefficient remaining
0.05 does not isolate a function-space effect.

The independently measured saved-group readout already shows median relative
block changes0.212091% for PE versus0.145991% for Large, with authenticated
initial/final fingerprints. It cuts against gross parameter under-movement,
without measuring useful functional adaptation. The reliable TRAIN-held
attribution still shows raw1024 PE starting ahead and ending behind; no
official quality or serving-speed number has changed.

Primary-source check: [Tensor Programs V](https://arxiv.org/abs/2203.03466)
describes widthwise transfer under a specified μP parametrization;
[Tensor Programs VI](https://arxiv.org/abs/2310.02244) explicitly discusses
limitations for deeper blocks such as transformers; the
[residual-depth scaling paper](https://arxiv.org/abs/2309.16620) does not turn
the heterogeneous pretrained PE/SigLIP2 comparison into a controlled scaling
family. The width-ratio×adapted-depth prediction is conjecture, not a verified
2.7× functional step or a qualified LR calibration. Only these primary abstracts
were independently checked; no claim of a full theoretical reproduction.

Dual critique `b5fcd8a39d1e40e6` completed normally: Opus
`76e2924679dd48ef` and Astra `cecde74cff044054` both **STOP**. Both independently
replayed the cheap counterexample. Their
[separate full reviews](evidence/compact_metric/sop-siglip2-substrate-v1/pe-update-geometry-v1/dual-result.json)
agree that thresholds/bootstrap uncertainty do not validate the statistic's
causal interpretation and that failure cannot establish capacity. The decision
does not adopt unverified universal optimizer-bound claims from either review.

No actual F0 displacement scores, multiplier, added epochs, Large rerun, corpus
probe, official evaluation or GPU job followed. The next step must use a
different supported mechanism rather than repair this gate with post-hoc
statistics or thresholds. Existing library/training work and protected Rust
changes are preserved. The production joint goal remains active and unmet.
