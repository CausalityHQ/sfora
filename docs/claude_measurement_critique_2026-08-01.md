# Claude CPU-measurement critique

Date: 2026-08-01. Read-only external review; no diagnostic or GPU run.

After the first external critique found no surviving method, a separate Claude
review was asked to design genuinely new CPU estimands over the five aligned CUB
HERD training packs. It inspected the existing analysis scripts and was told to
exclude measurements already made in the repository.

It proposed and then rejected five estimands:

1. an ecological turnover/nestedness decomposition of cross-class support sets,
   which would only select cross-class relations for transfer;
2. persistent-homology stability of the class-centroid graph, which would feed
   an occupied higher-order/global geometry operator;
3. invariant-causal-prediction tests across seeds, which would be a statistical
   mask for consensus distillation;
4. non-negative latent-factor rank of cross-class allocation, which would feed
   hierarchical/contextual class modelling; and
5. bimodality of per-image relation entropy, which would route samples through
   occupied hard-mining or loss-weighting machinery.

The review returned `NONE`: every positive diagnostic outcome mapped to an
operator already covered by candidates 1--58. This is useful negative evidence
about static final embeddings. Those packs can reveal new statistics, but any
statistic proposed so far remains a descriptor for selection, transfer, global
geometry, hierarchy, or routing. A new branch must therefore measure an object
not present in static `(embedding, label, example_id)` packs.
