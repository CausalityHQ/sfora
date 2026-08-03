# PFML reliability re-audit — 2026-08-03

## Scope and auditor

This audit was run after the operator warned that earlier conclusions could be
unreliable because of code defects and an incorrectly selected Claude model. The
reviewer was explicitly launched as `claude-fable-5` at high effort. It read the
primary PFML equations and supplement rather than inheriting the repository's
conclusions. This is adversarial review, not experimental evidence.

Primary sources: [CVPR paper and supplement](https://openaccess.thecvf.com/content/CVPR2025/html/Bhatnagar_Potential_Field_Based_Deep_Metric_Learning_CVPR_2025_paper.html)
and the [authors' project page](https://shubhangb97.github.io/potential_field_DML/).

## Verdict

No new executable error was found in the fixed PFML loss or training path. The
signs, kernel branches, sample/proxy populations, ordered pair counting, proxy
learning-rate multiplier, coupled Adam decay, warm-up semantics, and 16,200-step
Cars schedule were independently re-derived. This does **not** make the run an
exact reproduction.

The audit did find real defects in the supposedly fail-closed verification path:

1. the scalar analyzer pinned only a small subset of the recipe and did not pin
   the official train/test counts;
2. the PFML preset inherited a minimum-count filter instead of naming the full
   official partition;
3. balanced sampling could silently omit a short class while leaving its proxies
   in the objective;
4. the live-run guard depended on argument ordering;
5. the independent scorer assumed, without asserting, that every query had a
   same-class retrieval candidate.

These were repaired in commit `22b8223`. The analyzer now pins the complete
deciding-run configuration and exact Cars counts; future PFML presets request the
full official partition; proxy sampling rejects silent class exclusion; the guard
matches the unique report path; and the exporter rejects singleton classes.

The patch review itself initially returned **BLOCK**: the analyzer expected the
preset's dormant `train_steps=2000`, but the CLI serializes the epoch-resolved
`train_steps=16200`. The fixture copied the same wrong constant and therefore
passed. The value was corrected and a non-self-referential derivation from 8,054
examples, batch 100, and 200 epochs was added. The focused suite then passed 76
tests, Ruff, shell syntax, and a functional live-process refusal.

## Active-run provenance

The DGX deployment is a source copy rather than a Git worktree. Its trainer file
hash differs from current HEAD only because HEAD later wrapped two lines and
retracted a stale SPF docstring; a direct diff found no executable difference in
the loaded PFML path. The already-running report will correctly say
`dataset_selection_policy=legacy_minimum_filter`. It is accepted only if the report
also proves the exact official 8,054/8,131 example counts; because the legacy
operation is a pure filter, equality of counts proves that it removed nothing.
Future runs use `full_official_partition` explicitly.

## Remaining comparability limits

Any resulting number is a **single-seed fixed local interpretation**, not a
source-exact reproduction:

- the main paper specifies base learning rate `5e-4`, while the Cars supplement
  table specifies `1e-4`;
- selected `delta` and `alpha`, loss reduction, sampler, epoch convention,
  clipping, proxy normalization, and several initialization/augmentation details
  are not author-pinned;
- the run is nondeterministic and cannot establish the paper's five-run mean;
- raw best is test-selected at a ten-epoch cadence; final-epoch R@1 is primary.

If either preregistered metric gate fails, the run falsifies only this fixed local
interpretation. It does not falsify PFML, does not license post-hoc tuning, and its
field measurements cannot be used as provenance for a new mechanism.
