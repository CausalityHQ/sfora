# Iterative search protocol

Run this in a loop until a novel method that outperforms is found and benchmarked, or
until the evidence supports arguing that none exists. `pa_dual_ema` is candidate 1.

## Why a protocol rather than more ideas

Fifteen candidates failed and the failures were not random. Every one either

- **(a)** added regularisation to a base that was already fitting well, or
- **(b)** swapped the similarity function while leaving the supervision signal intact.

Three more died on **prior art** only *after* effort was spent. The loop below is ordered
to kill candidates cheaply, and in the order that kills most of them first.

## Idea generation: blind proposer, evidence-aware critic

Generation and judgement are deliberately separate roles. The proposer is a fresh,
repository-isolated Fable session. It receives only:

- the zero-shot image-retrieval problem and disjoint-identity evaluation;
- the legal training inputs and deployed-descriptor constraints;
- audited, matched-capacity numeric frontiers, with higher-capacity observations
  labelled separately. Every horizon must name backbone, deployed descriptor
  dimension, test-time operation, and material train-time machinery. A
  ResNet-50/2048-D GAP result is not a matched 512-D frontier merely because
  the backbone and cosine scorer agree; and
- the required form of a defensible answer: one mathematical training object, causal
  error mode, primary-source novelty case, decisive controls, frozen forecasts,
  falsifiers, cost, and a quantitative frontier-crossing argument.

The proposer receives **no repository measurements, candidate names, mechanism
shortlist, failure catalogue, proposed scientific direction, or previous model
answer**. Asking it to use augmentation, expanded supervision, another field, or an
"untried class" already chooses part of the answer and invalidates the independence of
the idea-generation pass. The prompt asks for exactly one concrete method or `NONE`.
An API failure, generic idea list, diagnostic without an executable method, or `NONE`
is not a candidate.

The proposer is constructive rather than dispositive. It must make a good-faith
algebra and novelty case and disclose risks, but it should not suppress its strongest
complete proposal merely because effect size is uncertain or neighboring work exists.
The mandatory fresh reviewer owns the adversarial `LIVE`/`DEAD`/`UNRESOLVED`
classification. This separation prevents cautious self-rejection from replacing the
required independent review; `NONE` remains appropriate only when no executable,
materially distinct method can be formulated.

Only after a complete proposal exists does the evidence-aware critic compare it with
the verified repository packet. Gate 1 asks whether an audited measurement actually
supports the proposed causal error mode; the proposer is not told that measurement in
advance. Gate 2 independently checks the algebra, searches primary literature
adversarially, and attacks the claimed distinction with the closest mechanism—not just
the proposed citations.

Before the local verdict, a **second fresh Fable session is mandatory**. Freeze the
first session's complete response verbatim and give the reviewer that frozen proposal,
the same legal-input/deployment constraints, the audited matched-capacity frontiers,
and an adversarial review rubric. Do not give it the proposer's conversation state,
the local critic's emerging verdict, or suggested repairs. The review must inspect:

- whether the mathematical objective actually has the claimed optimum and excludes
  the named shortcuts;
- whether the causal quantity is identifiable from the proposed measurements;
- closest primary prior art, including mechanism-equivalent work outside DML;
- whether controls distinguish the claimed mechanism from simpler occupied ones;
- whether forecasts follow quantitatively from measured premises and cross the proper
  matched-capacity horizon; and
- hidden data, compute, inference, tuning, and benchmark-protocol violations.

The reviewer returns `LIVE`, `DEAD`, or `UNRESOLVED` with cited reasons. It may reject
the frozen proposal but may not silently improve, reinterpret, or replace its training
object. Any substantive repair is a new proposal that must restart blind generation
and freezing. The local audit remains independent and authoritative only through
reproducible algebra, repository artifacts, and primary sources; agreement between two
language-model calls is not evidence by itself. Record both reviews and resolve every
material disagreement before Gate 3 or GPU work.

Failed proposals update the evidence and mechanism ledger, but are not summarized back
into the next proposer prompt. This prevents a sequence of nominally independent runs
from collapsing into local variations of the latest failure. The next pass uses the
same neutral problem specification in a fresh isolated session. Preserve the exact
prompt and complete proposal for every concrete candidate so proposal, audit, and any
later preregistration remain temporally distinguishable.

If a completed pass reveals that its supplied frontier mixed capacity lanes,
the output is non-authorizing even when it returns `NONE`. Correct the prompt
and rerun from a fresh session; repeated failure under mismatched frontier
arithmetic is not evidence that no method exists.

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

### Evidence levels and gate semantics (clarified 2026-08-07)

The gates are sequential controls, but they do not all have the same logical
force. A provenance or prior-art failure can invalidate a candidate before GPU;
an evidence failure limits the claim rather than proving that the mechanism is
impossible. Record the first failed gate and its evidence level explicitly:

* **Hard invalidity:** Gate 0 artifact/measurement failure, mechanism-equivalent
  prior art at Gate 2, or an implementation that does not match the frozen
  training object. These are `DEAD` unless the artifact or proposal is repaired
  and restarted as a new candidate.
  Gate 1 has two different outcomes and must not be collapsed into this bucket:
  an unmeasured or weakly supported premise is `UNRESOLVED`/`NO-GO` (no GPU),
  while a correctly recomputed measurement that falsifies the proposed causal
  signal is `DEAD` for that specific hypothesis. The latter closes the tested
  mechanism, not every method in its broad family.
* **Screen failure:** a preregistered forecast miss or a one-seed Gate-4 miss.
  This authorizes stopping expensive follow-up, but means “failed this matched
  screen,” not “the mechanism cannot work.” A near miss or uncertain paired
  control is `UNRESOLVED`, not forced `DEAD`.
* **Claim boundary:** Gates 5–7 (independent confirmation, raw/final reporting,
  and second-dataset replication) determine which claims are allowed. They do
  not erase a reproducible effect; without them the result is preliminary and
  cannot be called SOTA or general.

Adjacent prior art at Gate 2 is `LIVE-NARROW` with controls, not an automatic
death. A failed prediction should be recorded verbatim and not tuned away, but
the mechanism may remain scientifically unresolved when the run is underpowered
or the control/reference is uncertain. This distinction is required because
the repository has already retracted results for split, checkpoint-selection,
and BN-buffer errors; a strict binary gate can turn a measurement defect into a
false negative.

Write it into `docs/method_search_verdict.md` with its **mechanism**, not just its
number. That catalogue is worth more than any single arm.

Then run a fresh blind generation pass. The historical project hypothesis preferred
the one then-untried class—**change what supervision exists**, rather than how it is
scored—but this is evidence for the critic, not an instruction to seed the proposer.
Under the generation procedure above, Fable is not told this preference; it matters
only when judging whether a returned method repeats an exhausted mechanism class.

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

CRT (NeurIPS 2022) additionally reports **0.9448 In-Shop** with an
ImageNet-1K-pretrained MiT-B2/128-D single-view model. Its ResNet-50 ablation is
only 0.9238, so 0.9448 belongs to a higher-capacity transformer lane rather than
replacing the 0.939 comparable-CNN bar. A general, capacity-unrestricted
In-Shop SOTA claim must nevertheless confront it.

Our historical CUB best-over-test-training observations are **HIST 0.7082, Proxy
Anchor 0.6919**. Exact official pixels, partition, and scorer are now verified, but
the harness is modified and the artifacts cannot be independently rescored. The old
In-Shop **PA 0.9035 / HIST 0.9038** threshold is retracted because it used the wrong
image corpus. Corrected-pixel PA seed 0 is raw `0.9163`, final `0.9137`, which is one
seed—not a fixed gate. Future In-Shop screens require a same-seed, current-digest
paired control and retained final artifacts. Any general claim must confront at least
0.766 CUB, 0.949 Cars196, or 0.939 In-Shop under comparable capacity.

Commit and push after each gate. Report honestly whichever way each candidate goes.
