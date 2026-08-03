# Corrected In-Shop Proxy Anchor seed-1 reference preregistration

Date: 2026-08-03. Written and committed before seed-1 training. This is
reference infrastructure, not a method result or candidate screen.

## Why this run is needed

The only fully digest-bound corrected-pixel local reference is seed 0: raw
best-over-training R@1 **0.916303** and independently exported final R@1
**0.913701**. One seed cannot estimate variance, and the historical three-seed
0.9035 baseline used the wrong `img_highres` corpus plus a modified recipe. Every
future corrected In-Shop screen therefore needs another reference artifact
before a small effect can be interpreted.

## Locked run

Run seed 1 with unchanged recipe
`proxy_anchor.inshop.official-51db570`: official 256-pixel train/query/gallery
partitions, BN-Inception GAP+GMP, 512 dimensions, batch 180, AdamW, backbone and
head LR `6e-4`, proxy multiplier 100, weight decay `1e-4`, one warm-up epoch,
trainable BatchNorm, StepLR 20 / gamma 0.25, 60 epochs, value clipping at 10,
drop-last training, and raw Kaiming-normal proxies. Persist the final student
checkpoint and independently re-encode train, query, and gallery splits.

The command and output names are bound in
`scripts/run_inshop_corrected_reference_seed1.sh`. The run must execute from a
clean remote worktree at the committed `devbox/emafactorial` revision, not the
historically accumulated DGX directory.

## Prediction and failure conditions

- Expected raw best-over-training R@1: **0.916**, with the already registered
  acceptable interval **[0.907, 0.929]**.
- Expected final R@1: **0.914**, with a prospective integrity interval
  **[0.905, 0.923]**.
- The final checkpoint must be step 8,580, `final_training_state`, BN-Inception
  / 512-D, and config-identical to the report except seed 1.
- Official counts, identity separation, source paths, byte-content separation,
  and production/independent query-gallery R@1 must pass the existing exporter.

A metric outside either interval, any artifact/config mismatch, or scorer/corpus
failure triggers diagnosis rather than tuning. Even a passing seed supplies only
two observations; it does not restore the old 0.12-point sigma claim, authorize
a one-seed small-effect decision, or establish a standard deviation.

Estimated cost: about 2.2 GPU-hours for training plus final exports.

## Prelaunch deployment note

The historical DGX checkout has no configured Git remote, so `git fetch` could
not create the planned remote worktree. No training or output artifact had
started. The equivalent fail-closed deployment is therefore locked before
launch: create a new empty `/home/riomus/sfora-inshop-seed1` directory and rsync
only the paths emitted by local `git ls-files` at committed revision `fa5ed46`.
Protected untracked handoff/spec files, ignored reports, the dirty historical
checkout, and all local uncommitted state are excluded by construction. Record
hashes of the deployed recipe, trainer, data loader, runner, and exporter with
the result.
