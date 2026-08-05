# Pass-15 Gate-1 verified evidence packet

Date: 2026-08-05.

Purpose: compact input for the **local evidence-aware critic only** after a
blind proposal and its independent frozen-proposal review exist. Do not give
this packet, the repository, or its mechanism implications to the blind Fable
proposer or reviewer.

## Reliability boundary

The historical method ledger is not a commensurate experiment table. In-Shop
results from `img_highres`, SOP results using the wrong split or best-test
selection, the former leave-neighbour “selection correction,” and results
affected by established objective/buffer/dispatcher/evaluation defects are
quarantined. CUB and Cars results are not promoted merely because their
datasets were unaffected by the known In-Shop/SOP bugs. A new causal premise
must bind dataset membership, artifact selection, architecture, and metric to
independently recomputable or prospectively validated artifacts.

Authority: `docs/current_evidence_reliability_audit_321_2026-08-03.md` and
`docs/search_protocol.md`.

## Corrected In-Shop reference

The official 256-pixel retrieval corpus is functionally validated against the
Proxy Anchor authors' published BN-Inception checkpoint. Two local final
BN-Inception/512-D Proxy Anchor checkpoints are independently verified:

| seed | raw best-over-training R@1 | frozen-final R@1 |
| ---: | ---: | ---: |
| 0 | 0.9163032775 | 0.9137009425 |
| 1 | 0.9189056126 | 0.9167956112 |
| mean | 0.9176044451 | 0.9152482768 |

These are paired references, not a variance estimate. The legacy
`protocol="proxy-anchor-resnet50-512"` field is misleading; checkpoint keys,
`backbone_name`, recipe digest, and artifact metadata establish BN-Inception.

## Verified descriptive geometry

Independent float64 scorers reproduce very high training leave-one-out R@1:
`0.9950158411` for seed 0 and `0.9955933514` for seed 1. Therefore almost all
official-query error headroom is absent from ordinary leave-one-out training
identity retrieval.

For seed 0, nearest-foreign-image and nearest-foreign-proxy identities agree on
`0.1569044123` of training anchors. Error is `0.0238857424` conditional on
agreement versus `0.0014664772` conditional on disagreement. Seed 1 reproduces
the boundary: agreement `0.1259176261`, with error `0.0285363608` versus
`0.0014586925`. This is a descriptive error stratum, not proof that mining or
weighting it improves unseen-identity retrieval.

Across the two corrected checkpoints, 908 official queries fail under both;
the error-overlap coefficient is `0.7675401522`. Exact top-1 gallery-row
agreement is `0.8084822057`, but jointly wrong queries choose the same wrong
identity only `0.6475770925`. Thus query difficulty is persistent across two
initializations, while the winning impostor identity is materially less
stable. Two seeds still do not identify variance or a causal training signal.

## Verified augmentation-response relation

The corrected-corpus IPSR Gate-0 diagnostic exports a raw pack and is exactly
recomputed by an independent NumPy auditor that imports neither the production
IPSR code nor the training objective. Controlled transformation response is
not a disguised embedding-distance threshold:

| statistic | corrected value |
| --- | ---: |
| preference relations | 17,093 |
| anchor coverage | 0.660420 |
| eligible-class coverage | 0.773836 |
| closest-quartile same-class pairs rejected | 0.560879 |
| farthest-quartile same-class pairs accepted | 0.291475 |
| response-graph density | 0.361375 |
| multi-component eligible classes | 0.784654 |

The relation's existence and non-reduction to distance survive Gate 0. No
artifact establishes that enforcing this relation repairs official-query
errors or yields the `+2.530` points needed to reach the audited `0.939`
In-Shop CNN/GAP horizon. The former IPSR training result is retracted with the
wrong-pixel corpus. A new operator may cite the relation as an observed channel,
but its causal map and frontier-crossing forecast require separate evidence.

Authority: `docs/ipsr_corrected_corpus_gate0_result_2026-08-04.md`.

## What this packet does not support

It does not support an unseen-class EVT tail shift, hubness as an error cause,
low-rank shortcut monopoly, a shared cross-class nuisance basis, dataset-wide
variance estimates, selection-corrected R@1, or a general CUB/Cars/SOP causal
claim. Those premises require new prospectively specified measurements and
independent recomputation before Gate 1 can pass.
