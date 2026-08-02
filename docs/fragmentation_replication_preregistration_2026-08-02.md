# In-Shop fragmentation replication preregistration

Recorded on 2026-08-02 before training seeds 1 or 2 or exporting their
embeddings. This is a provenance diagnostic, not a method screen.

## Motivation

At the exact Proxy Anchor seed-0 epoch-10 operating point, 40.33% of eligible
In-Shop class 1-NN graphs were disconnected. After exact class-size
stratification, disconnected classes had leave-one-out R@1 **+3.534 points**
above connected classes. Many proposed methods treated that as evidence that
preserving within-class modes helps unseen-class retrieval, but the association
has never been checked under another initialization.

## Locked replication

Train the unchanged official In-Shop Proxy Anchor recipe for exactly 10 epochs
at seeds 1 and 2. Export the official training split's final normalized
embeddings. For each seed, run
`scripts/measure_spectral_class_connectivity.py` unchanged at temperature 0.1.
No test-set labels enter this diagnostic.

The run is bound to the same current recipe and code used for the seed-0
operating checkpoint. Only `train_epochs=10`, disabled periodic test evaluation,
the seed, and artifact paths are overridden. Report the exact emitted recipe
digest and hashes of both embedding packs.

## Predictions and falsification

Before observing either new pack:

1. each seed's eligible-class fragmentation fraction is predicted in
   **[0.35, 0.45]**;
2. each seed's exactly size-matched disconnected-minus-connected class-balanced
   leave-one-out R@1 is predicted to exceed **+2.0 points**;
3. the premise is falsified if either size-matched delta is **<= 0**, if the two
   new seeds disagree in sign, or if either pack differs from the locked
   dataset/recipe/epoch definition.

A pass establishes only that fragmentation is a reproducible marker. It does
not show causality and does not authorize diversity, subcentre, topology, or
pair-gating methods already killed at Gate 2. A successor still needs a distinct
supervision operator and prospective prediction.

Expected incremental cost after the active Cars reference: roughly 20 minutes
per seed plus CPU-only graph analysis.
