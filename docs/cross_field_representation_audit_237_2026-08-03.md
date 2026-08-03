# Cross-field representation audit 237

Date: 2026-08-03. Online prior art checked before implementation or GPU method
work. The corrected SOP baseline was still running and supplied no result to this
audit.

## Pattern separation from hippocampal biology

Dentate-gyrus models sparsify and orthogonalize similar inputs. Bird et al.,
*Robust and consistent measures of pattern separation based on information
theory and demonstrated in the dentate gyrus* (2024), show the central failure:
greater sparsity can improve classical separation scores while decreasing
information transmission. In a DML encoder the executable operations are top-k
activation, decorrelation, or a sparse autoencoder. Sparse coding, deep hashing,
uniformity/decorrelation, and reconstruction already occupy them. Adding an
information-preservation term does not create a new object; it is a sparse
autoencoder or information bottleneck. The repository's frequency intervention
already demonstrated the same destructive-separation failure empirically.

## Prototype--exemplar dual coding from psychology

Singh et al., *End-to-end Deep Prototype and Exemplar Models for Predicting Human
Behavior* (2020), and Zubek and Kuncheva, *Learning from Exemplars and Prototypes
in Machine Learning and Psychology* (2018), explicitly combine category
prototypes and stored exemplars. In DML this is a learned class proxy plus an
exemplar memory/XBM, with mixtures becoming multi-proxy models. Proxy Anchor,
memory-bank losses, Proxy Synthesis, SoftTriple, and Calibrate Proxy occupy every
executable variant.

## Hyperdimensional binding

Neubert and Schubert, *Hyperdimensional Computing as a Framework for Systematic
Aggregation of Image Descriptors* (CVPR 2021), already binds local image
descriptors and their positions into a single holistic vector and reports image
retrieval gains. Trainable binding becomes bilinear/tensor-sketch pooling; fixed
binding is the published HDC mechanism. The corrected repository evidence also
provides no provenance: the frozen Cars regional readout was `0.8159` versus
`0.8306` for the global vector.

## Verdict

**No candidate survives Gates 1 and 2.** These analogies reduce respectively to
sparse coding with information loss, proxy-plus-memory representation, and known
local-descriptor aggregation. No implementation or GPU run follows.

Primary sources:

- https://pmc.ncbi.nlm.nih.gov/articles/PMC10906873/
- https://arxiv.org/abs/2007.08723
- https://arxiv.org/abs/1806.01130
- https://openaccess.thecvf.com/content/CVPR2021/html/Neubert_Hyperdimensional_Computing_as_a_Framework_for_Systematic_Aggregation_of_Image_CVPR_2021_paper.html
