# Results consistency repair after selection-estimator retraction

**Status: stale claims corrected; no observed score changed.**

Candidate/audit 254 falsified the leave-neighbour peak-gap estimator as a selection
correction: noiseless curvature and endpoint slope generate positive gaps. Although
`docs/results.md` contained that retraction, later and historical sections still called
gap-subtracted values "selection-corrected," "selection bonuses," and evidence that
averaging was under-credited. The Cars196 RS@k reference repeated the same label after
the retraction section.

The results document now:

- labels the values as local peak-gap or invalid gap-subtracted diagnostics;
- withdraws the claimed EMA ranking reversal and protocol-under-credit mechanism;
- bases the In-Shop averaging and dual-EMA deaths on their registered raw conditions;
- labels TIRD's diagnostic without weakening its decisive raw failure; and
- reports Cars196 raw best and independently persisted final R@1, while retaining the
  local gap only as a non-causal historical diagnostic.

The same Cars wording was repaired in `docs/method_search_verdict.md`. No raw history,
final metric, paired test, candidate decision, or artifact was altered. Future claims
must use nested validation, a frozen epoch, or an independently selected/final
checkpoint to estimate selection effects.

