# UniCOM Rank-Finish Confirmation Plan

1. Extend the rank-finish runner under tests with an explicit finish seed,
   deterministic post-restore RNG reset, no early stop for confirmation seeds,
   and atomic inference-artifact publication.
2. Add a strict confirmation evaluator that authenticates the seed-0 screen and
   seed-1/2 receipts, recomputes every per-seed delta and the confirmation mean
   from seeds 1 and 2 only, and emits a canonical pass/fail receipt.
3. Run focused tests, Ruff, bytecode compilation, and the dependency-complete
   Python test suite; commit and push the frozen implementation.
4. Deploy the exact clean commit to DGX. Run seeds 1 and 2 serially from the
   registered epoch-4 checkpoint while actively monitoring the original
   process. Seed 2 uses metrics-only publication because it is robustness
   evidence and cannot become the release candidate.
5. Run the offline confirmation evaluator. If and only if it passes, execute one
   fixed seed-1 standard-test readout, recompute paired per-query evidence for
   control and candidate, and publish the paired quality receipt.
6. Record the terminal result, hashes, resource evidence, limitations, and
   release decision in the repository; run final verification and push.
