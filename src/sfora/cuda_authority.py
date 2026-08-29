"""Canonical CUDA device identity shared by FEPF runtime stages."""

from __future__ import annotations

import uuid


def canonical_cuda_device_uuid(value: object) -> str:
    """Return NVIDIA's canonical ``GPU-<uuid>`` device spelling."""

    raw = str(value)
    if raw.startswith("GPU-"):
        raw = raw[4:]
    try:
        normalized = str(uuid.UUID(raw))
    except (AttributeError, ValueError) as error:
        raise ValueError("CUDA device UUID differs") from error
    return f"GPU-{normalized}"
