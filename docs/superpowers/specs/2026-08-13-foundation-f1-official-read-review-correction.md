# Foundation F1 Official-Read Review Correction

**Status:** prospective correction. Commit `ad6d6b41cb75407d955460283614e76b6da815f5`
populated official-read registers but was not independently approved and has
not been executed. The registered receipt root remains empty and no official
pixel has been read.

**Corrects:**
`docs/superpowers/specs/2026-08-13-foundation-f1-official-read-addendum.md`.

## 1. Confirmed review defects

The original addendum relied on an operator checking out the intended commit,
but the runner did not authenticate that the populated registers were the
reviewed handoff. It also recomputed the train-only decision in the same process
that would consume official receipts without first requiring that decision and
split to equal the committed train-only evidence. Finally, repository-only
local fixture rows carried a tolerance but produced `passed=null`, making the
replay tolerance inert.

The train-only report remains valid historical evidence. Its status and values
are not changed. This correction authorizes no official read until the source
and authority sequence below has completed independent review.

## 2. Reviewed source commit `S_OFFICIAL`

Create one source commit descended from this correction that:

1. changes `src/sfora/foundation_pareto.py`, `src/sfora/cli.py`, and their
   focused tests only, while restoring the two official registers to their
   exact empty pre-authorization bytes;
2. adds strict v4 official-register schemas carrying, before `records`, exact
   fields `reviewed_source_commit`, `addendum_path`, `addendum_sha256`,
   `train_report_path`, `train_report_sha256`, `train_decision_sha256`, and
   `train_split_sha256`;
3. for official access only, authenticates the executing clean detached HEAD as
   a direct child of `reviewed_source_commit` whose sole changed paths are the
   two register files and `docs/foundation_metric_tolerances.json`;
4. authenticates the correction addendum Git blob/worktree SHA, the committed
   train-only report Git blob/worktree SHA
   `791cd1499327bd95abb5093d993a68c7192d44965af8669073bf35ad5b6ae066`,
   its `CONTINUE` status, decision
   `a3400169c6b94dbde2a2ecbb329a839ffba9edd43679587877133c5f3c83a9c8`,
   empty official arrays, and all three probe split hashes
   `7b075d601dbfa0b3f3587f80af169a378621cd4ba93aca35c2c9be745eac1f45`;
5. requires official execution arguments `validation_seed=0` and exact builtin
   float `validation_fraction=0.2` before any model or official-data load;
6. after recomputing the candidate/comparator train-only decision but before
   publishing an official receipt or loading official pixels, requires the
   current decision SHA and every current probe split SHA to equal the reviewed
   train-only bindings; and
7. converts an unavailable-native fixture into a separately named
   `repository_replay` audit whose pass predicate is exact
   `abs(repository_cosine - 1.0) <= tolerance`. Historical `unavailable/null`
   rows remain loadable only for old train-only evidence. A new screen stops on
   a false replay predicate.

The CLI must keep official access opt-in and must not derive authority from a
working-tree-only register. Source review must include RED/GREEN tests for a
wrong parent, merge/dirty checkout, extra path, addendum/report/SHA/decision/
split drift, wrong seed/fraction, recomputed decision drift before receipt
publication, contaminated-control access, repository-replay drift, and legacy
report readability.

## 3. Manifest-only authority commit `H_OFFICIAL`

After independent review finds `S_OFFICIAL` READY, create a direct child that
changes exactly these three files:

- `docs/foundation_test_read_register.json`;
- `docs/foundation_published_metric_register.json`; and
- `docs/foundation_metric_tolerances.json`.

Both registers use their v4 schema and bind exact `S_OFFICIAL`, this correction
path/SHA, the committed train-only path/SHA/decision/split, and the same exact
two arms, six metrics, order, purpose, and single-evaluation counts frozen in
the original addendum. The contaminated control remains absent. Change only
the contaminated control's `embedding_cosine` replay tolerance from `0.0` to
`0.000001`; retain candidate tolerance `0.000001` and identity-disjoint
comparator tolerance `0.0`. This value is frozen from the already observed
train-only replay error `7.485207631e-7`, before the official rerun, and is used
only to ensure the descriptive control does not block the candidate/comparator
screen on known deterministic-library drift. The comparator remains exact.

Independently review `H_OFFICIAL` as a manifest-only direct child, authenticate
all v4 bindings and the exact three-path diff, rerun the affected assurance,
and require no Critical or Important finding before execution.

## 4. Execution

The original `ad6d6b4` command is forbidden. Use the original addendum's exact
one-process command with `${H_OFFICIAL}` replaced by the reviewed manifest-only
commit. The runner itself performs the new source/authority/train-decision
pre-read checks. Receipt root, output, cache, and owned temporaries must be
absent; no official data may be opened by preflight. One completed official
read remains immutable and cannot be retried.
