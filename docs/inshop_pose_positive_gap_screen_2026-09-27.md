# In-Shop pose-positive mechanism screen, 27 September 2026

A read-only Fable consultation (`7911d3a4cfbe4777`) suggested training on
random gallery-role positive subsets to target In-Shop top-1 errors. Its
specific causal hypothesis was that removing a same-pose best positive would
account for at least **60%** of errors induced by thinning the held gallery.
This was a zero-training kill threshold, not a quality-selection rule. The
consultation proposed 200 seeded gallery draws and a stricter count: a removed
same-pose positive must be the only positive above the top impostor. This
probe instead reused one previously fixed gallery split and counted removed
best positives with the same filename pose. Its count is an upper bound on
that stricter event **for this split only**; it does not estimate the fraction
across the proposed draws. The older
[fixed asymmetric split](inshop_asymmetric_proxy_gate_2026-09-26.md) was
already rejected as a method-selection instrument, so this screen is
mechanistic and exploratory only.

The [source-bound probe](../scripts/probe_inshop_pose_positive_gap.py)
(SHA-256 `2f2ee1c5ea460eefe2eb0b34bb1eb889f66b26f53f15549958ba323b136175a8`)
re-exported the seed-179026 true-freeze checkpoint on the 12,599 official
In-Shop **TRAIN** held images. It verified each query's packed full-gallery
top-1 correct/incorrect bit against the original receipt; the receipt does
not contain winner identities or score vectors. It verified the older fixed
asymmetric 6,354-query/6,245-gallery row hashes, and applied the same packed
score and lower-ordinal tie rule. Pose means the filename suffix (`front`,
`side`, `back`, `full`, `flat`, or `additional`); it is metadata, not a learned pose
classifier. Each query's symmetric self-excluded gallery and the asymmetric
subset used the same encoded vectors.
The probe pins the historical training helper at SHA-256
`e2f7f8d16e2850a51aa85a3d3f4e04a80ee2a48681dcac55306fd792c8f19ba6`,
available as `scripts/train_sop_siglip2_compact.py` from Git commit
`e2ea109b`; replay must put that revision ahead of the current checkout on
`PYTHONPATH`, as in the DGX invocation. The current sibling helper differs
and correctly fails the probe's authority check.

| Fixed seed-179026 TRAIN diagnostic | Result |
| --- | ---: |
| Symmetric-gallery R@1 on the 6,354 asymmetric query rows | 98.3318% |
| Asymmetric-gallery R@1 on those rows | 97.6235% |
| New misses after removing gallery rows | 78 |
| Misses rescued by removing hard negative rows | 33 |
| New misses whose removed best positive had the same filename pose | **10/78 = 12.8%** |
| Encoder export wall on DGX Spark GB10 | 67.319 s |

Because the asymmetric gallery is a subset, any newly incorrect query must
have lost a positive that previously beat all available negatives; the probe
asserted this. The **12.8%** same-pose fraction is far below the proposed
**60%** threshold on this fixed split. Stop the pose-specific gallery-subset
loss before a full training run. The result does not rule out all positive-side learning losses,
and it says nothing about the official TEST quality or public latency.

The sole successful DGX service invocation was
`f77f951668124b99bb3011aabc449204`. Its [report](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-pose-positive-gap-179026-v2.json)
SHA-256 is `a64b428572c1cd94ce24e2d6dd9c5681af9aa971057cb0302fe9b3768b4b4f8a`;
its [journal](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-pose-positive-gap-179026-v2.journal.log)
SHA-256 is `60d39305bb14ef7d12a8dc349a44fe584b7a4681b4347d330783bf0ef3e69e8b`.
An earlier invocation failed before model loading because it compared the
older asymmetric receipt's 32-bit row hashes with the preflight's 64-bit
row digest. That [failed journal](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-pose-positive-gap-179026-v1-failed.journal.log)
SHA-256 is `5ff4fe041563e26d7a40592c23736a25cbafd3a0ac77e1a22fca71325506dafe`;
no quality was read from it.
