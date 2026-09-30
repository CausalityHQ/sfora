"""Isolated role-masked SmoothAP; no native/runtime import at module load.

Metadata is FIT-only, supplied in original bank ordinal order. No source,
head, optimizer, sampling, refresh, parameter, or serving change lives here.
The caller selects the arm and applies the unchanged external coefficient8.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib


GALLERY = 0
QUERY = 1


def build_role_inventory(labels, relative_paths):
    """Return (roles, padded positives) as tuples in original ordinal order.

    labels: sequence of nonempty string or integer product IDs (no bool).
    relative_paths: equally sized sequence of unique canonical relative POSIX
    strings, UTF8 encodable; no absolute/drive/dot/empty components or controls.
    Per product, sort SHA256(path UTF8); gallery takes the first
    max(1,min(n-1,round(n/2))) rows (Python ties-to-even rounding), query the rest.
    Singletons are gallery/invalid. Every positive row lists ALL same-product
    opposite-role original ordinals in ascending order, padded with suffix -1
    to the global maximum width (at least one column, including all singletons).
    Empty metadata returns ((), ()). No image, quality, or receipt is read.
    """
    if isinstance(labels, (str, bytes)) or isinstance(relative_paths, (str, bytes)):
        raise ValueError("role metadata must be sequences")
    try:
        labels, relative_paths = tuple(labels), tuple(relative_paths)
    except TypeError as exc:
        raise ValueError("role metadata must be sequences") from exc
    if len(labels) != len(relative_paths):
        raise ValueError("role metadata labels/paths differ")
    grouped = defaultdict(list)
    seen = set()
    digests = []
    for ordinal, (label, path) in enumerate(zip(labels, relative_paths)):
        if type(label) not in (str, int) or label == "":
            raise ValueError("role metadata product label differs")
        if (type(path) is not str or not path or "\\" in path or ":" in path
                or any(ord(char) < 32 or ord(char) == 127 for char in path)
                or any(part in ("", ".", "..") for part in path.split("/"))):
            raise ValueError("role metadata relative POSIX path differs")
        try:
            digest = hashlib.sha256(path.encode("utf-8")).digest()
        except UnicodeEncodeError as exc:
            raise ValueError("role metadata path is not UTF8 encodable") from exc
        if path in seen:
            raise ValueError("role metadata duplicate path")
        seen.add(path)
        digests.append(digest)
        grouped[label].append(ordinal)
    roles = [GALLERY] * len(labels)
    positives = [()] * len(labels)
    for indices in grouped.values():
        ordered = sorted(indices, key=digests.__getitem__)
        count = max(1, min(len(ordered) - 1, round(len(ordered) / 2)))
        gallery, query = sorted(ordered[:count]), sorted(ordered[count:])
        for i in gallery:
            positives[i] = tuple(query)
        for i in query:
            roles[i] = QUERY
            positives[i] = tuple(gallery)
    width = max((len(row) for row in positives), default=1)
    width = max(1, width)
    return tuple(roles), tuple(row + (-1,) * (width - len(row)) for row in positives)


def _validate_inputs(raw, bank, positive_ordinals, self_ordinals, *, labels, roles=None):
    """Independently validate descriptors, roles and complete positive inventory.

    Native-only. Bank stays unchanged/detached; validation never normalizes it.
    labels/roles are full-bank long vectors on the descriptor device. Role
    counts must match the metadata split, including singleton gallery. Hash
    membership is bound by the caller's metadata role identity, not guessed
    from descriptors. Return the validated nonempty-positive anchor filter.
    """
    import torch

    tensors = (raw, bank, positive_ordinals, self_ordinals, labels)
    if any(type(value) is not torch.Tensor for value in tensors):
        raise ValueError("bank rank inputs must be tensors")
    if (raw.ndim != 2 or bank.ndim != 2 or raw.shape[1] != 128 or bank.shape[1] != 128
            or not len(raw) or not len(bank) or not raw.is_floating_point()
            or bank.dtype != torch.float32 or positive_ordinals.ndim != 2
            or positive_ordinals.shape[0] != len(raw) or not positive_ordinals.shape[1]
            or self_ordinals.ndim != 1 or len(self_ordinals) != len(raw)
            or labels.ndim != 1 or len(labels) != len(bank) or labels.dtype != torch.long
            or positive_ordinals.dtype != torch.long or self_ordinals.dtype != torch.long
            or any(value.device != raw.device for value in tensors)
            or bool((self_ordinals < 0).any()) or bool((self_ordinals >= len(bank)).any())
            or bool((positive_ordinals < -1).any()) or bool((positive_ordinals >= len(bank)).any())):
        raise ValueError("bank rank geometry differs")
    with torch.autocast(device_type=raw.device.type, enabled=False):
        raw32 = raw.detach().float()
        raw_norm = torch.linalg.vector_norm(raw32, dim=1)
        bank_norm = torch.linalg.vector_norm(bank.detach(), dim=1)
        if (not bool(torch.isfinite(raw32).all()) or not bool(torch.isfinite(bank).all())
                or not bool(torch.isfinite(raw_norm).all()) or bool((raw_norm == 0).any())
                or not bool(torch.allclose(bank_norm, torch.ones_like(bank_norm), atol=1e-5, rtol=0))):
            raise ValueError("bank rank nonfinite/zero/nonunit descriptors")
    candidate_valid = torch.arange(len(bank), device=bank.device)[None, :] != self_ordinals[:, None]
    if roles is not None:
        if (type(roles) is not torch.Tensor or roles.dtype != torch.long or roles.ndim != 1
                or len(roles) != len(bank) or roles.device != bank.device
                or bool(((roles != GALLERY) & (roles != QUERY)).any())):
            raise ValueError("bank rank role vector differs")
        _, inverse, counts = torch.unique(labels, return_inverse=True, return_counts=True)
        gallery_count = torch.zeros_like(counts).scatter_add_(0, inverse, (roles == GALLERY).long())
        expected_count = torch.where(counts == 1, torch.ones_like(counts), torch.minimum(
            counts - 1, torch.maximum(torch.ones_like(counts), torch.round(counts.float() / 2).long())))
        if not torch.equal(gallery_count, expected_count):
            raise ValueError("bank rank per-product role split differs")
        candidate_valid = candidate_valid & (roles[None, :] != roles[self_ordinals, None])
    expected = candidate_valid & (labels[None, :] == labels[self_ordinals, None])
    valid = positive_ordinals >= 0
    if bool((valid[:, 1:] & ~valid[:, :-1]).any()):
        raise ValueError("bank rank padding must be a suffix")
    supplied = torch.zeros_like(expected, dtype=torch.long).scatter_add_(
        1, positive_ordinals.clamp_min(0), valid.long())
    if not torch.equal(supplied, expected.long()):
        raise ValueError("bank rank complete positive inventory differs")
    return valid.any(dim=1)


def _smooth_ap_masked_bank_loss(anchors, bank, positive_ordinals, candidate_valid, *, temperature):
    """Exact original SmoothAP arithmetic/order; only candidate mask is supplied.

    Validated FP32 unit anchors and full positives only; caller disables
    autocast, filters invalid anchors, and restores the fullbatch denominator.
    No rank clamp, approximation, diagonal mask, top-k, or truncation.
    """
    import torch

    valid = positive_ordinals >= 0
    scores = anchors @ bank.detach().T
    safe = positive_ordinals.clamp_min(0)
    positive_scores = scores.gather(1, safe)
    sigmoid_all = torch.sigmoid(
        (scores[:, None, :] - positive_scores[:, :, None]) * (2.0 / temperature)
    )
    candidate_rank = 0.5 + (sigmoid_all * candidate_valid[:, None, :]).sum(dim=2)
    positive_rank = 0.5 + (
        torch.sigmoid(
            (positive_scores[:, None, :] - positive_scores[:, :, None]) * (2.0 / temperature)
        )
        * valid[:, None, :]
    ).sum(dim=2)
    precision = positive_rank / candidate_rank
    return 1.0 - ((precision * valid).sum(dim=1) / valid.sum(dim=1)).mean()


def role_matched_bank_rank_loss(raw, bank, positive_ordinals, self_ordinals, *, labels, roles):
    """Return unweighted opposite-role SmoothAP with the fullmicrobatch divisor.

    raw: Bx128 live float descriptors; cast/normalize FP32 with autocast off.
    bank: Nx128 FP32 unit descriptors, detached internally and never changed.
    positive_ordinals: BxP long, ALL same-label opposite-role ordinals, suffix -1.
    self_ordinals: B long original bank ordinals, including repeated live rows.
    labels/roles: N long bank product IDs and GALLERY0/QUERY1 metadata roles.
    All tensors share a device. Temperature .01, no truncation, coefficient8
    external. Invalid anchors contribute graphzero, retaining denominator B.
    No baseline/native runtime imported; Torch is deferred until this call.
    """
    import torch
    from torch.nn import functional as F

    if roles is None:
        raise ValueError("bank rank role vector is required")
    keep = _validate_inputs(raw, bank, positive_ordinals, self_ordinals, labels=labels, roles=roles)
    with torch.autocast(device_type=raw.device.type, enabled=False):
        if not bool(keep.any()):
            return raw.float().sum() * 0
        anchors = F.normalize(raw[keep].float(), dim=1)
        candidate_valid = roles[None, :] != roles[self_ordinals[keep], None]
        loss = _smooth_ap_masked_bank_loss(anchors, bank, positive_ordinals[keep],
                                         candidate_valid, temperature=0.01) * (keep.sum() / len(keep))
        if not bool(torch.isfinite(loss)):
            raise ValueError("bank rank loss is nonfinite")
        return loss


def control_bank_rank_loss(raw, bank, positive_ordinals, self_ordinals, *, labels, original_loss):
    """Delegate the original unmodified smooth_ap_bank_loss, .01/no truncation.

    Tensor geometry is identical to the candidate; positives include ALL
    same-label nonself bank rows, independently validated (no role mask).
    original_loss must be the parent's original SmoothAP callable; this module
    does not import a frozen runtime closure. Preserve valid filtering before
    FP32 normalization, original reduction then valid.sum()/fullmicrobatch.
    Outer8 and actual arm selection remain caller-owned.
    """
    import torch
    from torch.nn import functional as F

    if not callable(original_loss):
        raise ValueError("original SmoothAP callable is required for control")
    keep = _validate_inputs(raw, bank, positive_ordinals, self_ordinals, labels=labels)
    with torch.autocast(device_type=raw.device.type, enabled=False):
        if not bool(keep.any()):
            return raw.float().sum() * 0
        loss = original_loss(F.normalize(raw[keep].float(), dim=1), bank,
                             positive_ordinals[keep], self_ordinals[keep]) * (keep.sum() / len(keep))
        if not bool(torch.isfinite(loss)):
            raise ValueError("bank rank loss is nonfinite")
        return loss


def bank_rank_loss(raw, bank, positive_ordinals, self_ordinals, *, arm, labels,
                   roles=None, original_loss=None):
    """Explicit arm dispatcher; caller supplies that arm's full positive table.

    arm='control': delegate original_loss with same-product/nonself positives.
    arm='role_matched': require roles with same-product/opposite-role positives.
    Route at the actual corrected terms boundary for every microbatch; do not
    infer an arm from rank_active or fall back on all-invalid microbatches.
    The parent owns CE+8*rank, source/RNG, optimizer, and detached bank refresh.
    """
    if arm == "control":
        return control_bank_rank_loss(raw, bank, positive_ordinals, self_ordinals,
                                      labels=labels, original_loss=original_loss)
    if arm == "role_matched":
        return role_matched_bank_rank_loss(raw, bank, positive_ordinals, self_ordinals,
                                           labels=labels, roles=roles)
    raise ValueError("bank rank arm must be control or role_matched")
