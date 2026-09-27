# In-Shop acquisition-group TRAIN falsifier, 27 September 2026

The rejected worst-positive loss leaves one distinct hypothesis: the current
representation overuses acquisition-local image evidence and misses unseen
products when the held gallery has no image from the query's acquisition
group. The July [acquisition audit](inshop_acquisition_audit.md) found a large
same-group cosine effect in an older model; it does not establish the effect
for the current SigLIP2 checkpoint. Here `group` means the filename's first
token, as parsed by the existing `_acquisition_series` helper. It is not a verified camera
or wearer identity.

## Frozen F0a decision, before reading the stratum

Use the existing seed-179026 true-freeze 1,000-update checkpoint and its
[expected-gallery TRAIN receipt](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-expected-gallery-179026/receipt.json)
(SHA-256 `d48e2c382fcfb7aa4807b00cc8b2a76fe098aebfa77334dbc2fd8402d36b46a4`).
Its 6,354 fixed query and 6,245 gallery roles are subsets of the official
In-Shop **TRAIN** partition, product-disjoint from 13,283 fit images, with
151 packed top-1 misses. Reconstruct roles using the existing metadata-only
helper and require the pinned partition, role and receipt hashes. For each
query, test whether its own product has a same-group image in the held gallery;
stratify the archived per-query packed hit flag. No encoder, scorer, official
query/gallery, or new quality selection is run.

Advance to F0b only if **at least 35% of the 151 misses** lie in the
no-same-group stratum **and** its miss rate is at least **2×** the same-group
stratum's miss rate. Otherwise stop the acquisition-target lane before any
training code or DGX GPU work. This rule tests enrichment, not a retrieval
improvement or a novel method. The zero-GPU source is
[`probe_inshop_train_acquisition_misses.py`](../scripts/probe_inshop_train_acquisition_misses.py).

If F0a passes, F0b must first re-export the pinned held embeddings and
recompute acquisition-group effects under the current checkpoint. The archived
pose-gap summary contains only aggregate induced misses, so its 78 induced
misses cannot be stratified from that JSON alone. Freeze a source-bound F0b
implementation and thresholds before reading its output. A later treatment
would need a matched TRAIN-only multi-seed quality and cost gate; F0a alone
does not authorize one. Keep the published official In-Shop 96.7% UNICOM
Recall@1 reference separate from this TRAIN stratum.
