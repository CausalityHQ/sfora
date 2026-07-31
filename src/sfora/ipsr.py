"""Preference construction for interventional principal-stratum ranking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class IPSRDiagnostics:
    preferred_indices: NDArray[np.int64]
    unknown_indices: NDArray[np.int64]
    anchor_coverage: float
    class_coverage: float
    mean_initial_loss: float
    preference_count: int


def build_ipsr_preferences(
    anchor_embeddings: NDArray[np.floating],
    signatures: NDArray[np.floating],
    labels: NDArray[np.integer],
    valid: NDArray[np.bool_],
    *,
    agreement_threshold: float = 0.5,
) -> IPSRDiagnostics:
    """Build the closest contradicted response preference for each anchor."""
    anchors = np.asarray(anchor_embeddings, dtype=np.float64)
    anchors /= np.maximum(np.linalg.norm(anchors, axis=1, keepdims=True), 1e-6)
    signatures = np.asarray(signatures, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    valid = np.asarray(valid, dtype=bool)
    count = len(labels)
    preferred = np.full(count, -1, dtype=np.int64)
    unknown = np.full(count, -1, dtype=np.int64)
    losses: list[float] = []
    eligible_classes = covered_classes = 0

    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        if len(members) < 3:
            continue
        eligible_classes += 1
        class_covered = False
        for anchor in members:
            peers = members[members != anchor]
            if not valid[int(anchor)]:
                continue
            agreement = signatures[peers] @ signatures[int(anchor)]
            similarities = anchors[peers] @ anchors[int(anchor)]
            compatible = valid[peers] & (agreement >= agreement_threshold)
            incompatible = valid[peers] & (agreement < agreement_threshold)
            if not compatible.any() or not incompatible.any():
                continue
            incompatible_rows = np.flatnonzero(incompatible)
            # np.argmax is stable and peers follow training-row order, fixing ties.
            u_row = incompatible_rows[int(np.argmax(similarities[incompatible_rows]))]
            u_similarity = float(similarities[u_row])
            contradicted = compatible & (similarities < u_similarity)
            if not contradicted.any():
                continue
            compatible_rows = np.flatnonzero(contradicted)
            p_row = compatible_rows[int(np.argmax(similarities[compatible_rows]))]
            preferred[int(anchor)] = int(peers[p_row])
            unknown[int(anchor)] = int(peers[u_row])
            losses.append(float(np.logaddexp(0.0, u_similarity - similarities[p_row])))
            class_covered = True
        covered_classes += int(class_covered)

    preference_count = int((preferred >= 0).sum())
    return IPSRDiagnostics(
        preferred_indices=preferred,
        unknown_indices=unknown,
        anchor_coverage=preference_count / count if count else 0.0,
        class_coverage=covered_classes / eligible_classes if eligible_classes else 0.0,
        mean_initial_loss=float(np.mean(losses)) if losses else 0.0,
        preference_count=preference_count,
    )
