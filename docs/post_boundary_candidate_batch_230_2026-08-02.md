# Candidate batch 230: token architecture and data construction

Date: 2026-08-02. Status: **no candidate survives Gate 2**. No diagnostic,
implementation, or GPU.

Boundary 229 left token/spatial supervision and training-data construction
outside its output-Gram representation argument. Six cross-field proposals were
generated with Claude and attacked before implementation.

| candidate | executable operator | provenance | mechanism death |
| --- | --- | --- | --- |
| 230a, count-sketch token pooling | hash and sum spatial-token outer products into the deployed 512-d vector | no matched positive measurement; see correction below | Compact Bilinear Pooling (Gao et al., CVPR 2016) is Tensor Sketch pooling; NetVLAD, Fisher, GeM, and bilinear pooling occupy the aggregation family. |
| 230b, MIL negative-bag suppression | for a different-class pair suppress its maximum cross-token match, without a positive regional loss | no matched positive measurement; see correction below | max-instance bag supervision and attention MIL are established; the retrieval form is DIML/DeepEMD. Region Proxy Anchor's measured CUB 0.6466 additionally points against training the local evidence. |
| 230c, Fourier amplitude intervention | swap/randomize amplitude spectra across labelled images while retaining phase and class label | In-Shop acquisition cosine .8199 within versus .6396 across, 90.9% same-acquisition neighbours, 7.18x gap amplification | FDA (Yang and Soatto, CVPR 2020), FACT (Xu et al., CVPR 2021), and APR (Chen et al., ICCV 2021) already use label-preserving Fourier amplitude interventions. |
| 230d, description-length pruning | estimate a sample's conditional coding contribution and prune or resample redundant images | acquisition-series redundancy | description-length selection was closed at candidate 127; deployed action is data pruning/sampling, adjacent to forgetting/GraNd/EL2N. |
| 230e, morphological token support | require identity evidence to form a compact connected spatial support | fine-grained parts | attention compactness, erasing, and part discovery are established by MA-CNN, ADL, and extensive part-based fine-grained recognition. |
| 230f, coded image superposition | feed linear multiplexes of two training images and require demultiplexable class evidence | none beyond generic support expansion | combination-weighted targets are Mixup/Metrix; the candidate also fails Gate 1. |

The strongest near-miss is 230c because its mechanism directly targets the
largest measured shortcut at approximately 1x cost without dataset metadata.
It dies on novelty, not feasibility. It could only be presented as a
benchmark-transfer evaluation of a known method, analogous to weight averaging;
that does not satisfy the standing objective and no GPU is spent on it.

## Corrections to the independent audit

Two proposed general closures are **not adopted**.

First, the deployed head's `avgpool -> Linear` identity does not make spatial
tokens redundant during training. A position-dependent loss applies different
gradients to individual backbone tokens; in general no loss of the pooled vector
produces the same token Jacobian. Cross-token correspondence is not the only
possible token-level escape. What is supported is narrower: the six token
operators proposed here are occupied or empirically unpromising.

Second, “preserve, mix, destroy, split, or code the label” is a useful taxonomy
of the constructions tried here, not an exhaustive theorem about every possible
data-derived target. A construction can also introduce structured, relational,
or latent supervision; those cases must be audited individually. Boundary 229
therefore remains evidence-bounded rather than becoming an impossibility claim.

The honest result is that this batch found no survivor. Spatial/token
supervision remains a logical escape class, but a successor must introduce a
specific measured spatial relation and distinguish itself from DIML, DeepEMD,
part discovery, attention regularization, masked prediction, and cross-view
completion before GPU use.

## Provenance correction after the batch

The `+6.67` CUB points used while generating 230a/230b were initially described
as a frozen regional-feature gain. That was wrong. They are the improvement
from repairing the **evaluation metric of an already-trained `region_pa` arm**:
fixed-slot concatenated cosine `0.5775` to MaxSim `0.6442`. The arm still lost to
Proxy Anchor (`0.6466` mean versus `0.6825`). The actual frozen-feature probe is
on Cars: global pooling `0.8306`, MaxSim `0.8159`, **-1.47 points**. Therefore the
repository does not currently contain positive evidence that a token aggregator
beats the global descriptor. This correction weakens Gate 1 for 230a and 230b;
their prior-art deaths remain unchanged.
