# Pass 174 — Morphogenetic Embedding Head (MEH), CPU preregistration

## Motivation

The corrected In-Shop evidence shows reproducible class fragmentation but no
useful algebraic-connectivity signal. The remaining architecture hypothesis is
that a shared local iterative update can preserve multiple within-class modes
without selecting pairs or adding extra data. Split the 512-D descriptor into
64 eight-dimensional cells and apply four residual updates using one shared
local rule over each cell and its two cyclic neighbours:

`h_g^{t+1} = h_g^t + eps*tanh(W0 h_g^t + W- h_{g-1}^t + W+ h_{g+1}^t + b)`.

Concatenate and L2-normalize at deployment. Proxy Anchor remains the training
supervision; the update rule is the only changed object. This is inspired by
neural cellular automata/morphogenesis, not by a pair-loss analogy.

## Gate-2 distinction

Neural Cellular Automata Manifold (CVPR 2021) and later NCA work use recurrent
local rules for morphogenesis or image tasks, but I found no benchmark-matched
single-descriptor DML evaluation using a shared NCA head. This is
`LIVE-NARROW`, pending a full primary-source audit if the CPU screen survives.

## CPU gate

On the corrected In-Shop training pack, hash-split each identity into support
and query halves. Train MEH plus class proxies on frozen support embeddings and
compare with a parameter-matched linear 512→512 head and the raw descriptor.
MEH must beat the linear control by at least `0.5` R@1 points on held-out
queries and have a non-degenerate effective rank (participation ratio ≥80).
Failure kills the architecture before GPU.

## GPU preregistration if admitted

One In-Shop seed, full BN-Inception/512 training, expected frozen-final R@1
`0.9190` versus `0.9153889`; falsified below `0.9175`. A matched linear-head
control and a no-iteration MEH ablation are mandatory.

## CPU result

On a deterministic 300-class In-Shop support/query subset, a frozen-backbone
MEH proxy trained with class proxies reached held-out support-to-query R@1
`0.996667` versus `0.998333` for the matched linear 512→512 control (−0.167
points). MEH's participation ratio was `95.17` versus `86.84`, so it expanded
rank but did not improve retrieval. The preregistered `+0.5`-point Gate-1
criterion failed; no GPU implementation is authorized.
