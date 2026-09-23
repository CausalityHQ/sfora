"""Authenticated optional projection profiles for reproduced external encoders."""

from __future__ import annotations

import hashlib
from importlib.resources import as_file, files

from sfora.compact_metric import CompactMetricEncoder

_SOP_PROFILE_NAME = "oml_vits16_sop_power_whitening_128.sfora"
_SOP_PROFILE_FILE_SHA256 = "d34e263fc79a3df4391c3972311fcafd6216c10388346769b16eaa8fd8936cc2"
_SOP_PROFILE_ENCODER_SHA256 = "07e6e0f38dae4d509fe1e27b10aa650acda1f08d799faa025793cf7686a78cd4"


def load_oml_sop_compact_encoder() -> CompactMetricEncoder:
    """Load the 384-to-128 head for the official OML ``vits16_sop`` checkpoint.

    The external OML image encoder is not bundled. Its reproduced checkpoint
    SHA-256 is ``2701830538f31bd2dabb06622475cc889b1095580fc57218d0293e7a6bce53a7``.
    The returned head produces signed-int8-128 codes for the 130-byte packed
    search wire. This profile is specific to Stanford Online Products.
    """

    resource = files("sfora").joinpath("model_artifacts", _SOP_PROFILE_NAME)
    with as_file(resource) as path:
        if hashlib.sha256(path.read_bytes()).hexdigest() != _SOP_PROFILE_FILE_SHA256:
            raise ValueError("OML SOP profile artifact differs")
        encoder = CompactMetricEncoder.load(path)
    if encoder.sha256 != _SOP_PROFILE_ENCODER_SHA256:
        raise ValueError("OML SOP profile encoder differs")
    return encoder
