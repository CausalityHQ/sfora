# In-Shop three-way partition and embedding audit

Date: 2026-08-03

## Finding

The independent In-Shop final exporter checked official row/class counts,
query/gallery ID and path disjointness, and equality of their test identity sets. It did
not reject duplicate IDs or paths **within** a split, and it did not prove that the
training identities, IDs, and paths were disjoint from both query and gallery. Thus a
malformed three-way partition could retain the expected counts while leaking a training
identity or duplicating rows.

It also trusted the encoder's returned label order and normalized output without
explicitly verifying either before scoring.

## Repair

Before encoding, `verify_official_partition` now requires:

- exact `(rows, identities)` of train `(25882, 3997)`, query `(14218, 3985)`, and
  gallery `(12612, 3985)`;
- unique example IDs and resolved source paths within every split;
- zero train/query and train/gallery identity overlap;
- identical query/gallery identity sets; and
- zero ID and resolved-path overlap for every split pair.

After encoding, the exporter requires exact returned-label order, finite embeddings,
and unit norms for query and gallery. The retrieval JSON records the complete partition
audit. Four focused tests cover the independent cosine scorer, zero norms, a valid
three-way partition, and duplicate-ID rejection; Ruff and whitespace checks pass.

The repaired exporter was deployed before the queued In-Shop run could start. Together
with exact checkpoint/report config binding, this makes the next In-Shop reference
fail closed on the artifact and dataset failures found in the historical line.
