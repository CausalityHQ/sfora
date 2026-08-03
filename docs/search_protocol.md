# Iterative search protocol

Run this in a loop until a novel method that outperforms is found and benchmarked, or
until the evidence supports arguing that none exists. `pa_dual_ema` is candidate 1.

## Why a protocol rather than more ideas

Fifteen candidates failed and the failures were not random. Every one either

- **(a)** added regularisation to a base that was already fitting well, or
- **(b)** swapped the similarity function while leaving the supervision signal intact.

Three more died on **prior art** only *after* effort was spent. The loop below is ordered
to kill candidates cheaply, and in the order that kills most of them first.

## The gates — stop at the first one a candidate fails

**0. Validate the motivating measurement from artifacts.** Configuration intent,
code inspection, a plausible image count, and a green unit test are not sufficient.
Before a number can motivate a candidate, independently verify the exact dataset
membership, artifact labels and selection state, evaluation self-exclusion, and metric
recomputation wherever the persisted artifacts permit it. If they do not permit
recomputation, label the claim code-derived and rerun/export what is missing. The
2026-08-03 SOP audit found both a wrong benchmark split and training embeddings saved
at the best-test epoch; all historical SOP conclusions were retracted. A bug can
invalidate a negative just as easily as a positive.

The current evidence boundary is enumerated in
`docs/current_evidence_reliability_audit_321_2026-08-03.md`.  Historical verdict
entries are not a commensurate empirical table: use only the explicitly verified
packet for new provenance.  In particular, the legacy `protocol` family string
does not establish which backbone executed; bind architecture claims to the
checkpoint, resolved `backbone_name`, recipe digest, and state keys.

**1. Provenance of the idea.** It must be motivated by a measurement *in this repo*, not
by an armchair analogy. `pa_dual_ema` is the model: it came from the factorial showing
the distillation-target role wants 0.999 while the evaluated-average role wants 0.99, so
a single EMA must lose one. Write the motivating number down.

**2. Prior art, before any GPU.** Search properly. H3 was EMAN (Cai, CVPR 2021).
Gauge-fixing was Hoffer (ICLR 2018) and Pernici (TNNLS 2022). Snapshot-Procrustes was
Huang (ICLR 2017) + Izmailov (UAI 2018). Multi-center was SoftTriple (ICCV 2019) + SwAV.
**All four were found after the work.** If it exists, record the citation in
`docs/method_search_verdict.md` and move to the next candidate. Cheap to check,
expensive to skip.

**3. Pre-register.** A number you expect, and the condition that falsifies it, committed
*before* the deciding run. Four have been registered here and all four failed — that is
the process working.

**4. Screen on corrected In-Shop, not CUB.** The historical claim that In-Shop
σ = 0.12 point and one seed is decisive came from the wrong `img_highres` corpus and
is retracted. Establish a paired corrected-corpus reference first. A single seed may
kill a large miss against a prospectively fixed threshold, but cannot establish a small
gain or a variance model. Use `pa_ema_avg_bnfix`-style buffer handling wherever
BatchNorm is trainable, or the average is only half an average. CUB screening at n=1
produced a false positive here once already (+0.52, retracted).

**5. Confirm out-of-sample.** Never quote the seeds that generated the hypothesis.
In-sample gave +0.890; out-of-sample +0.427. An sd from 2–3 runs is worthless: n=3 gave
0.153, n=6 gave 0.367.

**6. Report raw and independently selected/final metrics.** The historical output of
`scripts/measure_selection_bias.py` is only a local-neighbour peak-gap diagnostic, not
an identified selection correction: noiseless curvature and endpoint slope produce a
positive gap. Report it only under that name. A defensible correction requires nested
validation, a frozen epoch, or an independently selected checkpoint; never relabel the
local trend as corrected R@1. See `docs/selection_bias_estimator_retraction_254_2026-08-03.md`.

**7. Replicate on a second dataset.** Cars196 or CUB. One dataset is an observation; two
is a result.

## When a candidate dies

Write it into `docs/method_search_verdict.md` with its **mechanism**, not just its
number. That catalogue is worth more than any single arm.

Then generate the next candidate, preferring the one untried class: **change what
supervision exists**, rather than how it is scored.

## Standing fact corrected 2026-08-01: the general benchmark ceiling is occupied

IDEAL's 72.3 remains non-comparable: its HIST baseline is 69.7, below HIST's published
71.4 and our 70.82 six-seed reproduction, and it uses four-view inference. However,
*Potential Field Based Deep Metric Learning* (Bhatnagar and Ahuja, CVPR 2025) was missed
by the earlier audit. It reports standard single-view ResNet-50/512-D results over five
runs: CUB 73.4 ± 0.3, Cars196 92.7 ± 0.3, and SOP 82.9 ± 0.2. Its 15 proxies/class and
200-epoch recipe require care in a reproduction, but its evidence is credible enough that
the old ~0.715 CUB ceiling is occupied. See `docs/recent_dml_horizon_scan_2026-08-01.md`.

Two later primary-source checks close more of the horizon. VAPNet (NeurIPS 2023)
reports standard-split, single-model ResNet-50/GAP results of **0.762 CUB, 0.948
Cars196, and 0.939 In-Shop**. AdvRF (ICCV 2025) reports **0.766 CUB and 0.949
Cars196** with the same broad evaluation form and does not test In-Shop. Both use
2048-D embeddings and 200 epochs and report no seed count or uncertainty. Those
limitations prevent significance claims about small differences, but they do not
leave the general benchmark regions open. See
`docs/open_set_fg_retrieval_horizon_2026-08-01.md`.

Our historical CUB best-over-test-training observations are **HIST 0.7082, Proxy
Anchor 0.6919**. Exact official pixels, partition, and scorer are now verified, but
the harness is modified and the artifacts cannot be independently rescored. The old
In-Shop **PA 0.9035 / HIST 0.9038** threshold is retracted because it used the wrong
image corpus. Corrected-pixel PA seed 0 is raw `0.9163`, final `0.9137`, which is one
seed—not a fixed gate. Future In-Shop screens require a same-seed, current-digest
paired control and retained final artifacts. Any general claim must confront at least
0.766 CUB, 0.949 Cars196, or 0.939 In-Shop under comparable capacity.

Commit and push after each gate. Report honestly whichever way each candidate goes.
