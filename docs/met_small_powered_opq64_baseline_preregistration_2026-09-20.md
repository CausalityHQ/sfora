# MET-small powered OPQ64x8 baseline preregistration

## Question

Before training another rank-aware codec, measure the compression loss of the
strongest already-observed 64-byte baseline on the powered 3,050-query
protocol. The exact float ceiling on that protocol is known, but prior OPQ64x8
measurements used the 129-query full-gallery validation protocol and cannot be
pooled with the reduced-gallery proxy.

## Frozen measurement

- authenticate feature SHA-256
  `0277f717e73416e9cb7d1a8d57c3db08e05aea20cf9803f1fcedbfc502b81105`;
- construct the unchanged deterministic 3,050-query / 35,257-gallery split;
- normalize all source rows;
- use Faiss 1.12.0 with one OMP thread;
- fit exactly `OPQ64_768,PQ64x8` on the reduced gallery only;
- require exactly 64 code bytes for every gallery item;
- add the same gallery to the trained codec and use its native exhaustive
  squared-L2 ADC search, not decoded-vector scoring;
- compare against exhaustive float squared-L2 search on the identical gallery;
- report mMP@5 and Recall@1 on the powered proxy as primary, and on all 129
  shifted validation queries as descriptive secondary evidence;
- report fit/search time and gallery reconstruction MSE.

There is one deterministic arm and no parameter selection. The official MET
test remains sealed.

## Frozen interpretation

Set `codec_near_ceiling=true` only if float-minus-OPQ64x8 is at most `0.005`
mMP@5 and at most `0.01` Recall@1 on the powered proxy. A pass means further
64-byte scorer/codebook complexity has little room on this representation and
the research priority moves upstream to embedding/projection quality and
downstream to index performance. A failure establishes material codec
headroom and authorizes one preregistered rank-aware codec experiment.
