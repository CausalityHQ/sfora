# Gallery-linking proposals (contextless specialist 9993eeead2054764, Sonnet 5.5 fallback, 2026-10-31)

Input: docs/gallery_linking_brief_2026-10-31.md only. Full text is in the consult receipt. The specialist's ranking:
1. **D1 provenance kernel:** JPEG tables, native size, noise and white balance as a second pair kernel plus a two-hop
   bridge. Its estimate: +0.6 × 0.25.
2. **D4 set-context listwise re-ranker:** a set transformer over {q} ∪ top-32. Its estimate: +0.3 × 0.15.
3. **D3 cluster-union token coverage:** union-MaxSim over high-purity clusters. Oracle ceiling first, needs ≥ +1.0.
4. **D2 typed competition/hubness normalisation:** s' = cos − β·ρ(c)·(1+γ·detail(c)). Its estimate: +0.15 × 0.25. 15 min.

Root decisions:
- **D1 REJECTED on validity.** It exploits eBay upload-session encoding artefacts, not product appearance. That does
  not survive a normalised production catalogue, and it is dead on In-Shop. The specialist itself rated it GRAY.
- **Run D2 and the D3 oracle ceiling first.** Both are cheap and label-free at index time, built leave-query-out.
- **Run D4 only if they leave headroom.**
Shared harness:
- 5,851 held queries, scored against two galleries: the full TRAIN gallery and a held-only gallery.
- Fitted knobs chosen by 5-fold CV over held products.
- Paired product bootstrap.
