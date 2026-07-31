# Candidate 8: rival-signature positive graph (RSPG)

Status: gates 1–3 passed with qualified novelty; In-Shop graph diagnostic
passed; seed-0 screen running.

## Gate 1 — provenance: PASS

Five independent CUB **training** embedding packs show that top-5 same-class
neighbours have a mean +0.0913 advantage in the correlation of their rankings
over negative-class centroids. Within-class embedding similarity and rival
profile similarity correlate 0.7048. The effect is present in every seed. At
the same time, global class-centred pseudo-modes are unstable (cross-run ARI
0.06–0.07), and sub-centre Proxy Anchor lost about 1.7 pt. The measured signal
is local agreement about *which other classes are rivals*, not a stable subclass
centre.

## Gate 2 — prior art: QUALIFIED PASS

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

None found makes *agreement over negative-class identities* the source of a new
positive relation. A close same-class pair can fail RSPG and a distant pair can
pass, so it is not merely easy-positive mining. Novelty confidence remains
medium and must be described that way.

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

## Gate 4a — fixed graph diagnostic: PASS

On the In-Shop seed-0 run (recipe digest `7a57b992b6a4`), the epoch-10 graph had
**13,209 / 153,115 edges**, density **0.0863**, and **0.8703** of eligible
classes had two or more connected components. Both preregistered conditions
passed without changing the thresholds. Training therefore continued to the
registered R@1 decision; this diagnostic alone is not a method win.
