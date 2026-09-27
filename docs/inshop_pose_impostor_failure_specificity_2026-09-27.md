# In-Shop pose-impostor failure specificity, 27 September 2026

The first metadata-only screen on the fixed 6,354-query/6,245-gallery
official-TRAIN held roles found same-pose top impostors on 61 of 151 misses,
versus 29.1502 expected from gallery pose frequencies: 2.0926×, above its
frozen 2× threshold. Its [raw receipt](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-pose-impostor-f1-179026.json)
has SHA-256 `a495f7ed55ade83aade714ec80a8c28e0f4634308a59a7ad11cecf695adf0b31`.
The result is exploratory: nearest visual neighbors do not follow gallery pose
frequencies. Five of six `flat`-pose misses are same-pose against only 0.074
expected; excluding `flat` lowers enrichment to 1.926×.

Before running another encoder export, the independent Opus 5.5 and GPT-6
Astra review identified the necessary failure-specific comparison. The frozen
[probe](../scripts/probe_inshop_pose_failure_specificity.py) will replay the
same checkpoint, scorer, stable ties and 151 archived miss-impostor pairs,
then find the best eligible negative for every query. It compares same-pose
top-impostor prevalence between misses and hits within query-pose strata,
excluding `flat` from the primary Mantel-Haenszel odds ratio. A seeded
2,000-draw product-cluster bootstrap supplies a 95% interval. Report `flat`
and every other pose separately. **Stop the pose-conditioned training lane if
the lower interval bound is at most 1.3.** A pass only authorizes further
diagnostics; it does not validate training, a method-selection panel, or an
official result. The earlier 20-role-draw and k=2 selector suggestion is
rejected because it reuses the previously rejected held proxy.

This probe uses one full export on the existing DGX Spark GB10 under the shared
GPU lock. The exact source, model, checkpoint, split and miss receipt are
checked before scoring. No production path changes on this diagnostic.

## Terminal result

The sole DGX unit `sfora-inshop-pose-failure-specificity-v1.service`
(invocation `2a870fb000414a79b133408c4b601847`) exited 0. The
[result](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-pose-failure-specificity-179026.json)
SHA-256 is `cd6121cbbff59238a67512e4ab64c6569e0093344008d0b94e03a39a1e535710`;
the [journal](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-pose-failure-specificity-179026.journal.log)
SHA-256 is `7bec27ad37325bb06c1be2447e6b48ad84d22106b91643c0888fafa7c8d3648a`.
The run replayed the prior 151 misses and their exact top-impostor row ordinals.
An independent aggregate calculation reproduced the stratified odds ratio.

| Fixed TRAIN held 6,354-query/6,245-gallery roles | Same-pose top impostor | Different-pose top impostor |
| --- | ---: | ---: |
| Misses, excluding flat | 56 | 89 |
| Hits, excluding flat | 1,671 | 4,456 |
| Misses, flat only | 5 | 1 |
| Hits, flat only | 25 | 51 |

The flat-excluded query-pose-stratified miss odds ratio is **1.729**;
the seeded product-cluster bootstrap 95% interval is **[1.225, 2.432]**.
Its lower bound fails the frozen **>1.3** requirement. Export wall was
**67.373 s** on DGX Spark GB10; training, image-to-top-k latency and
production quality were not measured by this diagnostic.

**Decision:** stop the pose-conditioned bank-negative lane before loss code,
training or an official TEST read. The original gallery-frequency enrichment
was real on this proxy but did not provide sufficiently strong
failure-specific evidence. The rejected 20-draw/k=2 selector remains rejected.
