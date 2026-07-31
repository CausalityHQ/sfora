# Supervision candidates: prior-art pre-screen

Search performed 2026-07-31, before implementation. I searched supervised DML,
unsupervised DML, clustering/pseudo-label, local-neighbourhood, graph/label-propagation,
positive-mining, multi-proxy, and intra-class-variance work. The ranking criterion is
`P(genuinely distinct) × P(In-Shop R@1 ≥ 0.9085)`. Rank 1 was initially recommended
for GPU; the post-screen CPU diagnostic below killed it before a valid GPU result.
Ranks 2–3 are explicit prior-art casualties. None is now recommended for GPU.

## 1. Rival-signature positive graph (RSPG) — **DEAD: CPU graph too dense**

Post-screen update: the fixed CPU-only diagnostic retained 64.49% of
within-class pairs, above the preregistered 60% ceiling. The multi-component
fraction was exactly 25%. RSPG therefore fails before a valid GPU screen; the
thresholds were not tuned.

The dataset-scope defect was then adjudicated conservatively by rerunning on an
independently exported, CUDA-disabled In-Shop training pack. That graph failed
in the opposite direction: density 0.0040, below the 0.05 minimum, with 0.9997
multi-component classes. The known epoch-10 partial-run density of 0.0863 makes
the representation-stage dependence explicit, but is not a valid result. RSPG
remains dead and no ablation is activated.

1. **Mechanism** — Warm up the ordinary Proxy Anchor model for 10 epochs, then make one
   stop-gradient, full-training-set embedding pass. For each image \(i\), form a *rival
   signature* \(q_i\): a distribution over **other-class identities**, obtained by
   softmaxing the nearest distance from \(i\) to each other class (keep the nearest 32
   rival classes). For two images with the same ground-truth class, create a positive
   edge only when their rival signatures have top-8 overlap of at least 4 and
   Jensen–Shannon divergence at most 0.25; give the edge target
   \(1-\mathrm{JS}(q_i,q_j)\). Same-class pairs that fail the gate are unknown, not
   negatives. Replace Proxy Anchor's indiscriminate positive term with a cosine
   supervised-contrastive term over these edges; retain its ordinary different-class
   negatives. Refresh signatures once at epoch 40 from an EMA snapshot. Thus the added
   supervision says “these two members of a class share the same discriminative
   context,” and is derived only from training images, labels, and the model being
   trained. It is neither a new inference similarity nor an auxiliary regularizer on
   the old all-same-class signal: it replaces that signal with a new positive relation.

2. **Motivating measurement** — The project owner observes a “spiky” space in which an
   image can be closer to images of another class than to its own class representative.
   RSPG uses the *identity and agreement of those spikes* as signal rather than merely
   treating them as hard negatives. The other directly relevant project measurement is
   that fixed sub-centres hurt badly (CUB R@1 0.675), so the candidate deliberately
   creates pair supervision without centres, occupancy constraints, or subclass IDs.
   Before training, the go/no-go CPU diagnostic is that at least 25% of classes have
   two or more connected components under the registered thresholds and that edge
   density is neither below 5% nor above 60% of within-class pairs; otherwise the
   proposed signal is absent or nearly identical to the class label.

3. **Closest prior art, with citation** — The closest sample-mining work is Wang et al.,
   [Online Soft Mining and Class-Aware Attention
   (AAAI 2019)](https://arxiv.org/abs/1811.01459), which weights same-class pairs by
   their **pairwise embedding distance**, explicitly favouring local/easy positives.
   Easy Positive independently selects each anchor's nearest same-class example
   ([Xuan et al., *Reducing Class Collapse in Metric Learning with Easy Positive
   Sampling*, NeurIPS 2020 workshop](https://openreview.net/forum?id=QQzomPbSV7q)).
   Classical learned-neighbourhood metric learning jointly learns target-neighbour
   assignments, but again from instance quality and within-space neighbourhood
   structure ([Wang, Woznica & Kalousis,
   2012](https://arxiv.org/abs/1206.6883)). ProxyGML builds a sample/proxy similarity
   graph and reverse-propagates the **known class labels**, using multiple proxies per
   class ([Zhu et al., *Fewer is More*, NeurIPS
   2020](https://proceedings.neurips.cc/paper/2020/hash/ce016f59ecc2366a43e1c96a4774d167-Abstract.html)).
   RSPG's target is instead equivalence of two samples' distributions over *negative
   class identities*. A close pair can be rejected and a distant pair accepted; it
   creates no proxies, clusters, pseudo-class IDs, or propagated class prediction.
   I found contextual graphs and negative-class mining separately, but no primary DML
   paper that turns agreement of cross-class rival signatures into positive edges.
   The remaining mandated checks do not supply this supervision either: SoftTriple and
   sub-center ArcFace are multi-centre classifiers; BIER and ABE split/ensemble
   embeddings; DiVA adds complementary self-supervised heads; DFML routes a factorized
   backbone; SwAV balances prototype occupancy; HIER learns a hyperbolic class hierarchy;
   and Roth, Vinyals & Akata import language. That is a specific distinction, not proof
   of novelty; an expert search should be repeated before publication.

4. **Pre-registered prediction** — In-Shop R@1 **0.9100** (one seed), versus Proxy
   Anchor 0.9035 and HIST 0.9038. It is falsified if R@1 is **< 0.9085** (failure to
   clear +0.5 point over Proxy Anchor), or if the pre-training graph diagnostic above
   fails; thresholds must not be tuned on the test set.

5. **Cost** — About **1.05× Proxy Anchor**, estimated **5.3 GPU-hours if the local
   baseline is 5 GPU-hours**: two no-gradient training-set passes plus cheap signature
   construction, no extra model and no external encoder. Report actual wall time beside
   the baseline because this repository contains no measured In-Shop GPU-hour datum.
   Inference is unchanged: one model, one view, one 512-d vector, cosine retrieval.

## 2. Local-connectivity positives — **DEAD**

1. **Mechanism** — Build a mutual-\(k\)-nearest-neighbour graph separately inside each
   training class from EMA embeddings; create positives only on its edges, adding the
   minimum spanning-tree edges needed to keep each class connected. Non-edges within a
   class are unknown. This replaces the all-pairs class signal with locally derived
   must-link supervision.

2. **Motivating measurement** — It directly addresses multi-modal classes and avoids a
   proxy in empty space; the project's 0.675 sub-centre result suggests connectivity
   without centres would be the least damaging form. No project measurement, however,
   shows that local distance is a trustworthy positive oracle.

3. **Closest prior art, with citation** — **DEAD.** This is the deep, graph-refreshed
   version of target-neighbour metric learning. LMNN already pulls selected nearest
   same-class target neighbours while excluding impostors ([Weinberger & Saul, JMLR
   2009](https://www.jmlr.org/papers/v10/weinberger09a.html)); Wang, Woznica & Kalousis
   explicitly learn the target-neighbour assignments iteratively
   ([2012](https://arxiv.org/abs/1206.6883)); Easy Positive selects the nearest
   same-class positive; and OSM continuously favours nearby positives to preserve
   intra-class manifolds. MST connectivity is an engineering choice, not a new source
   of supervision sufficient to distinguish the method.

4. **Pre-registered prediction** — If it were run, In-Shop R@1 **0.9070**; falsified
   below **0.9085**. It should not be run because it fails the prior-art gate.

5. **Cost** — Approximately **1.05× baseline** (about **5.3 GPU-hours if baseline is
   5**), with periodic embedding passes. Inference would remain single-model,
   single-view, 512-d.

## 3. Discovered within-class pseudo-subclasses — **DEAD**

1. **Mechanism** — Cluster each training class's EMA embeddings, select \(K\) by a
   stability criterion, and use `(class, cluster)` pseudo-labels as positives while
   treating different clusters of the same class as unknown. The pseudo-subclasses are
   derived only from training images and refreshed during training.

2. **Motivating measurement** — The single-proxy/multi-modal mismatch motivates it, but
   the project's direct sub-centre Proxy Anchor result is **0.675 CUB R@1**, substantially
   worse than the ordinary Proxy Anchor reproduction (0.6946), which is already strong
   negative evidence.

3. **Closest prior art, with citation** — **DEAD.** SoftTriple learns multiple centres
   per class and softly assigns samples to them ([Qian et al., ICCV
   2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Qian_SoftTriple_Loss_Deep_Metric_Learning_Without_Triplet_Sampling_ICCV_2019_paper.html));
   sub-center ArcFace uses multiple sub-centres per class
   ([Deng et al., ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/1445_ECCV_2020_paper.php));
   deep clustering DML already converts learned clusters into pseudo-label supervision
   ([Nguyen et al., GCPR 2021](https://arxiv.org/abs/2009.04091)). Hard clustering and
   suppressing cross-cluster attraction do not create a defensible distinction, and the
   local experiment already rejects the mechanism's premise.

4. **Pre-registered prediction** — If it were run, In-Shop R@1 **0.9000**; falsified
   below **0.9085**. It should not be run because both prior art and local evidence kill
   it.

5. **Cost** — Approximately **1.08× baseline** (about **5.4 GPU-hours if baseline is
   5**) for embedding passes and per-class clustering. Inference could remain one view
   and 512-d if the clusters were training-only, but that does not rescue the candidate.
