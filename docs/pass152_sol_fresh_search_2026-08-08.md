# Pass 152 — fresh independent Sol search (NONE, 2026-08-08)

Fable and Claude were unavailable at their weekly limit, so a new Codex Sol
consultation was used as a cold fallback.  It was read-only and did not use the
GPU.  The prompt included Pass151 and the two preregistered pixel diagnostics,
but asked for exactly one new train-time object or `NONE`.

## Result

`NONE`.  Gate 1 cannot be defended.  CIS dropout produced only `+0.056` raw
and `+0.042` final R@1 points; RGB/edge and HOG-like residuals produced
incremental held-out-category Delta AUCs of `-0.000029` and `-0.000007`; and
the available In-Shop labels map one-to-one to product identities.  Stable
test-error persistence is evaluation-derived, while training fragmentation is
compatible with approximately `0.9955` training leave-one-out retrieval and
does not identify underfitting.

Gate 2 also closes executable uses of the remaining measured signals:

* opposing gradients reduce to PCGrad/CAGrad-style gradient surgery;
* class-disjoint assessment reduces to DML-ALA or pair weighting/mining;
* fragmentation reduces to SoftTriple, graph, or neighbourhood learning;
* cross-run consensus is distillation;
* transferable variation is Meta Variance Transfer, and the repository's
  transport diagnostic is adverse.

No mathematical training object, CPU diagnostic, corrected In-Shop threshold,
or GPU experiment is authorized without inventing an unmeasured premise.  This
is a bounded negative, not proof that future data-derived signals are
impossible.  Full evidence is in Pass149, Pass150, and Pass151.
