# MET-small equal-byte OPQ geometry result

## Result

Redistributing the same 64-byte payload across more, narrower product
subquantizers and fewer bits per subquantizer does not jointly improve
MET-small validation mMP@5 and R@1.

| Codec | mMP@5 | R@1 | reconstruction MSE | fit/eval seconds |
|---|---:|---:|---:|---:|
| OPQ64×8 | **0.718992** | 0.736434 | **0.00018280** | 311.05 |
| OPQ128×4 | 0.709044 | 0.744186 | 0.00026389 | 306.43 |
| OPQ256×2 | 0.707106 | **0.767442** | 0.00053774 | 405.61 |

All codes store exactly 64 bytes and were fit with one Faiss CPU thread on the
same 38,307 normalized fit rows. Finer-partition/lower-bit codes improve
top-one accuracy but reduce top-five neighborhood quality and increase
reconstruction error. All three Faiss arms retain 768 dimensions; this result
is not evidence about representation rank. Neither lower-bit arm satisfies the
frozen joint rule, so
`higher_rank_lower_bit_supported` is false.

This is a multi-objective tradeoff rather than evidence that either metric is
wrong.  A successor must preserve fine local ordering while retaining broader
high-rank evidence; changing only the uniform bit allocation is insufficient.

The deterministic one-thread OPQ64 mMP is `0.008140` above the earlier
unrestricted-thread fit from the prospective validation.  Those are different
fitting contracts and must not be pooled.  Future codec evidence must bind
threading and deterministic fitting explicitly.

## Authority

- result:
  `docs/evidence/rank_finished_l14_336_met_small_equal_byte_opq_geometry_v1.json`
- result SHA-256:
  `755fbbb0f7d7415bc82c3e5714b82a126d72cd9175e410f3a9eb35b8b49d2742`
- driver SHA-256:
  `28781acbd5053e0ba1af889f66df61da4b3c54e24466853334d7e88353e50a86`
- Faiss `1.12.0`, one OMP thread, exact 64-byte code shapes.
