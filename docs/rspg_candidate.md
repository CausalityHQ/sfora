# Candidate 8: rival-signature positive graph (RSPG)

Status: **DEAD at Gate 4.** Contaminated-path In-Shop seed 0 reached raw best
R@1 0.8452 versus the registered 0.9085 minimum; seed 1 became logically
unnecessary and no ablation ran. See `docs/method_search_verdict.md`, candidate
18, for the self-erasing-graph mechanism.

## Gate 1 — provenance: PASS

Five independent CUB **training** embedding packs show that top-5 same-class
neighbours have a mean +0.0913 advantage in the correlation of their rankings
over negative-class centroids. Within-class embedding similarity and rival
profile similarity correlate 0.7048. The effect is present in every seed. At
the same time, global class-centred pseudo-modes are unstable (cross-run ARI
0.06–0.07), and sub-centre Proxy Anchor lost about 1.7 pt. The measured signal
is local agreement about *which other classes are rivals*, not a stable subclass
centre.

## Gate 2 — prior art: LIVE, narrowly

The exact candidate is narrower than the response-distillation/feature-mask
proposal rejected as candidate 7. RSPG does not reproduce a response vector and
does not learn a conditional metric. It uses agreement between two same-class
images' distributions over negative-class identities to decide whether that
pair is a positive edge; failed pairs become unknown.

Closest primary-source mechanisms checked before implementation:

- OSM weights positive pairs by their direct embedding distance
  ([Wang et al., AAAI 2019](https://arxiv.org/abs/1811.01459));
- learned target-neighbour metric learning updates neighbourhood assignments
  from instance quality and within-space geometry
  ([Wang, Woznica & Kalousis, 2012](https://arxiv.org/abs/1206.6883));
- ProxyGML propagates known labels over a sample/multi-proxy graph
  ([Zhu et al., NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/ce016f59ecc2366a43e1c96a4774d167-Abstract.html)); and
- contextual similarity distillation preserves a teacher's neighbourhood
  relations
  ([Wu et al., CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Wu_Contextual_Similarity_Distillation_for_Asymmetric_Image_Retrieval_CVPR_2022_paper.html)).

Two especially close training-signal precedents bound the claim. Contextual
Similarity Distillation makes contextual descriptors a training signal
([Wu et al., CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Wu_Contextual_Similarity_Distillation_for_Asymmetric_Image_Retrieval_CVPR_2022_paper.html)).
Contextual Similarity Optimization explicitly optimizes contextual similarity
as a supervised metric-learning loss
([Liao et al., arXiv:2210.01908](https://arxiv.org/abs/2210.01908)).

RSPG excludes the target class, describes rivals by class identity rather than
instance kNN membership, and gates positive-to-unknown edges. The adversarial
audit found this complete operator unoccupied, narrowly. The novelty claim lives
or dies on the gate and requires the three controls in
`docs/rspg_ablation_plan.md`; a headline retrieval gain alone is insufficient.

## Gate 3 — preregistration

The deciding screen is In-Shop seed 0, never CUB. Prediction: **0.9100 R@1**
against the existing 0.9035 three-seed Proxy Anchor mean and 0.9038 HIST mean.
The candidate is falsified if raw best R@1 is **<0.9085**.

Before the positive-edge loss is enabled, the fixed training-only graph
diagnostic must also pass: at least 25% of classes have two or more connected
components, and retained edge density is between 5% and 60% of within-class
pairs. Thresholds are fixed at top-8 rival overlap ≥4 and Jensen–Shannon
divergence ≤0.25 over the nearest 32 rival classes. A diagnostic failure kills
the candidate without threshold tuning or test evaluation.

One warm-up of 10 epochs precedes graph construction; the graph refreshes once
at epoch 40 from a stop-gradient EMA snapshot. Training data, labels, and the
current model are the only sources. Inference remains one model, one view,
512-dimensional cosine retrieval.

## CPU go/no-go diagnostic: FAIL

The required standalone diagnostic was run with `CUDA_VISIBLE_DEVICES=""` on
the existing training-only `herd_tt_seed0.train.npz` pack. It retained **109,375
/ 169,596 edges**, density **0.6449**, above the preregistered maximum 0.60.
The multi-component fraction was **0.2500**, exactly the minimum. The conjunction
therefore failed: under fixed thresholds the graph is too close to the original
all-same-class relation. Thresholds were not changed.

This kills RSPG on CUB at the fixed operating point. It did not validly decide
the planned In-Shop screen because the registered task omitted the diagnostic
dataset. The corrected decision reruns the unchanged rule on fixed In-Shop
training embeddings, with the contaminated decision path disclosed.

No prior In-Shop retrieval result or selection correction is valid. The
precommitted consequence of a pass would have been main-screen seeds 0 and 1
before any claim, with no Cars run or ablation before both cleared the decision.

### One-step In-Shop diagnostic: INVALID FOR THE OPERATING POINT

**This decision path is contaminated because the favourable epoch-10 partial-run
graph was already known before option (b) was chosen.** With thresholds unchanged,
the independently materialized CPU-only In-Shop pack
`inshop_pa_cpu_step1.train.npz` retained **606 / 153,115 edges**, density
**0.0040**, below the registered 0.05 minimum; its multi-component fraction was
0.9997. The pack contains 25,882 × 512 embeddings and has SHA-256
`ad43b4037f30c07f0353d87bdf38849a77745a8534f7423f5f09155b8e0e4601`.
It used the official Proxy Anchor initialization after exactly one
CPU optimizer step because no completed In-Shop checkpoint existed. CUDA was
disabled for both export and graph construction.

This number does not decide RSPG. A one-step model is effectively the pretrained
BN-Inception backbone plus a nearly random embedding head; its rival signatures
do not represent the learned In-Shop class structure encountered when the gate
actually fires at epoch 10. Treating it as a clean correction would answer the
wrong representation-stage question.

The adjudicated operating-point diagnostic therefore trains **plain official
Proxy Anchor for exactly 10 In-Shop epochs**, with periodic test evaluation
disabled, exports final epoch-10 training embeddings, and applies the unchanged
graph rule on CPU. This costs GPU time because the supposedly free diagnostic
depends on a trained representation. The already-seen partial-run density 0.0863
makes the confirmation path contaminated and must remain adjacent to its result.

### Exact epoch-10 operating-point diagnostic: PASS

**This confirmation is contaminated because the prior partial run had already
revealed a favourable epoch-10 graph.** The independently rerun official
Proxy Anchor epoch-10 pack retained **13,253 / 153,115 edges**, density
**0.0866**, and had multi-component fraction **0.8735**. It passes the unchanged
0.05–0.60 and ≥0.25 gates and closely reproduces the invalidly observed
0.0863/0.8703. The 25,882 × 512 training pack has SHA-256
`85e76245603689c824ec3f6aefceb67eee34fb7df94d3a825977a8bd4d139b27`.

The corrected gate therefore warrants full RSPG In-Shop seeds 0 and 1. Both are
required because of contamination. No ablation, Cars run, or claim is allowed
until the two raw screens are judged against 0.9085.

## Procedural correction

An In-Shop seed-0 process was mistakenly started before this CPU-first task file
was read. It was terminated after epoch 10 and is excluded from evidence. Its
in-training graph happened to have
**13,209 / 153,115 edges**, density **0.0863**, and **0.8703** of eligible
classes with two or more components, but it cannot retroactively satisfy the
registered ordering and produced no deciding retrieval result. No artifact from
that partial run may be quoted.

The required asymmetry test now exists and passes: the gate rejects a close
same-class pair with disjoint rival signatures and accepts a distant pair with
identical signatures. This confirms that the implementation did not collapse
to Easy Positive or OSM distance mining, but it does not rescue the failed
free diagnostic.
