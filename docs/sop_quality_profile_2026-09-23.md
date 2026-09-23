# Optional SOP quality profile

Sfora includes a strict 384-to-128 projection profile for the independently
reproduced OML `vits16_sop` image encoder. It is a dataset-specific quality
profile, not a general replacement for the fit-only selector and not a claim
of scientific state of the art. The external image model is not bundled.

| Official Stanford Online Products split | mAP@R | Recall@1 | Stored gallery bytes |
| --- | ---: | ---: | ---: |
| Prior Sfora UNICOM-L/14 compact path | 0.511490 | 0.770173 | 128 code / 130 served |
| OML ViT-S/16 float-384 source | 0.654393 | 0.865575 | 1,536 descriptor |
| OML plus fitted power-whitening int8-128, exact packed score | 0.641825 | 0.859757 | 128 code / 130 served |

The OML compact profile improves Sfora's previous absolute SOP result by
+0.130335 mAP@R and +0.089584 Recall@1 at the same search width and serving
bytes. It loses 0.012568 mAP@R and 0.005818 Recall@1 against its own float
source, so it should not be described as better than OML itself. The
projection's training-only three-fold choice and the observed official test
are in `docs/evidence/compact_metric/oml-vits16-sop-power-whitening-v1.json`.
That earlier scorer normalized the integer codes in float32. A new
independent replay used the exact served f16 inverse norms and produced
0.6418248620 mAP@R / 0.8597567023 Recall@1 from the bundled artifact. Its
canonical receipt is
`docs/evidence/compact_metric/oml-vits16-sop-packed-profile-verification-v1.json`,
SHA-256 `780805d2a090a6faa048cd3b5ba39c1a692fc494641bbfa987f509c9a9f7193c`.
The 0.000031 mAP difference is score arithmetic, with identical Recall@1.
The official split was already observed in prior work; all these local scores
are `claim_eligible=false`.

The exact external checkpoint is the OML `vits16_sop` ViT-S/16 at SHA-256
`2701830538f31bd2dabb06622475cc889b1095580fc57218d0293e7a6bce53a7`,
from [OML's model zoo](https://github.com/oml-team/open-metric-learning).
Use its 224-pixel `get_normalisation_resize_hypvit` transform, single-scale
normalized 384-D output, and official SOP image split. The profile loader is:

```python
from sfora.model_profiles import load_oml_sop_compact_encoder

profile = load_oml_sop_compact_encoder()
packed = profile.encode_packed(oml_features.cpu().float().contiguous())
```

The bundled profile is
`src/sfora/model_artifacts/oml_vits16_sop_power_whitening_128.sfora`,
SHA-256 `d34e263fc79a3df4391c3972311fcafd6216c10388346769b16eaa8fd8936cc2`.
Its canonical encoder SHA-256 is
`07e6e0f38dae4d509fe1e27b10aa650acda1f08d799faa025793cf7686a78cd4`.
The source fit checkpoint was SHA-256
`47fa92323a18154973d417be965346e198fc14e81008dd7d3c83d8a4bc5182b9`.
The loader verifies both strict artifact framing and the pinned hashes.

The existing 128-D native search kernel accepts this wire without changing
its storage or arithmetic shape. That does not alone establish equal tail
latency on a million OML gallery codes; a paired public-call replay using
this profile's code distribution is required before a performance release
claim. Encoder runtime is separate: OML ViT-S/16 at 224 pixels and the prior
UNICOM ViT-L/14 at 336 pixels have not yet been paired in one latency replay.
For published quality context, [UNICOM Table 4 (ICLR 2023)](https://arxiv.org/pdf/2304.05884)
reports 88.8%, 89.9%, and 91.2% SOP Recall@1 for supervised ViT-B/16,
ViT-L/14, and ViT-L/14@336 respectively. Its table's 88.0% figure is the
*previous* frontier, not its own strongest result. These full-width supervised
systems remain ahead of this profile's 85.98% and use different training and
deployment budgets. Raising a compact profile to that quality range requires
an adapted image backbone and matched encoder-latency evidence.
