# Pass 85 — Multi-Threshold Substitutability DML scope review

The search found a mechanism-level escape from the closed label-only families:
replace identity equivalence with uniformly sampled human judgments at several
strictness levels asking whether two images are mutually acceptable visual
substitutes. A cumulative-link likelihood would train a single cosine
descriptor; inference remains an ordinary 512-D vector.

The Gate-1 motivation is the corrected four-seed positive-transfer gap
(-0.04968 mean), which shows that identity equivalence is too coarse for
unseen identities. Primary neighbours include active oracle querying, human
similarity datasets, relative-attribute supervision, and post-hoc alignment,
but none combines fixed multi-threshold substitutability labels with end-to-end
zero-shot DML in this exact form.

This cannot be admitted to the current autonomous GPU loop: it requires new
human annotations and changes the benchmark’s information channel from
label-only DML to annotation-augmented DML. Generating the labels from the
existing identities would collapse it back to already-closed pair supervision.
No implementation or GPU run occurred. It is retained as an explicit scope
boundary and a possible future study, not claimed as a result.
