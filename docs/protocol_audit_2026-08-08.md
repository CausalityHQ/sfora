# Search-protocol audit (2026-08-08)

The recent run of pre-GPU `NONE` results is plausibly process-induced.  The
blind proposer was forbidden from seeing repository measurements, while Gate 1
required a proposal to be motivated by one.  This makes provenance nearly
impossible unless the proposer guesses a known measurement.  A universal
`+0.05` pixel-diagnostic threshold further rejects small but reproducible signals
that a training operator might amplify.  These are discovery-gate defects, not
evidence that the DML design space is exhausted.

The repaired loop adds a second, independent measurement-conditioned proposer
lane.  It sees one verified measurement packet and must produce a falsifiable
operator; it does not inherit any blind proposal or verdict.  Gate 1 now asks for
artifact-reproducibility and causal relevance, with an effect threshold registered
per diagnostic rather than a universal AUC floor.  The blind lane remains to
preserve genuinely unsuggested novelty.

Gate 2 is repaired to require mechanism equivalence: the cited work must match
the training object, data flow, and decision point.  Shared vocabulary such as
“gradient”, “dynamical”, or “physics-inspired” is only adjacency.  Adjacent work
gets `LIVE-NARROW` and a component-removal ablation rather than automatic death.

This is a process correction, not a positive result.  Existing negatives remain
valid where their own artifact or matched-run falsifier failed; the correction
reopens candidate generation only.
