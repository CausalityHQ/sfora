# Candidate 8: rival-signature positive graph (RSPG)

Status: CPU go/no-go diagnostic failed; candidate dead; no valid GPU screen.

## Gate 1 — provenance: PASS

Five independent CUB **training** embedding packs show that top-5 same-class
neighbours have a mean +0.0913 advantage in the correlation of their rankings
over negative-class centroids. Within-class embedding similarity and rival
profile similarity correlate 0.7048. The effect is present in every seed. At
the same time, global class-centred pseudo-modes are unstable (cross-run ARI
0.06–0.07), and sub-centre Proxy Anchor lost about 1.7 pt. The measured signal
is local agreement about *which other classes are rivals*, not a stable subclass
centre.

## Gate 2 — nearest prior art; novelty not claimed

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
instance kNN membership, and gates positive-to-unknown edges. The asymmetry test
establishes that this is not direct distance mining, but it does not establish
novelty over contextual training. RSPG must not be recorded as novel;
`docs/rspg_prior_art.md` records the narrow unresolved boundary.

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

The candidate stops here. No In-Shop result, selection correction, CUB screen,
or Cars run is valid or warranted.

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
