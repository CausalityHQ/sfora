"""Leakage-safe primitives for the preregistered TSPA mechanism screen."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from sfora.kernels.set_maxsim import fused_set_maxsim

F1_TRAIN_CLASSES = frozenset(range(49))
F1_VALIDATION_CLASSES = frozenset(range(49, 82))


def _observed_classes(labels: torch.Tensor, *, name: str) -> frozenset[int]:
    if labels.ndim != 1 or labels.numel() == 0:
        raise ValueError(f"{name} labels must be a nonempty vector")
    if labels.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{name} labels must use an integer dtype")
    return frozenset(int(value) for value in labels.tolist())


def validate_f1_class_partition(
    *,
    train_labels: torch.Tensor,
    validation_labels: torch.Tensor,
) -> None:
    """Require the exact disjoint Cars train-class bands frozen for F1."""

    if _observed_classes(train_labels, name="training") != F1_TRAIN_CLASSES:
        raise ValueError("F1 training must contain exactly Cars classes 0 through 48")
    if _observed_classes(validation_labels, name="validation") != F1_VALIDATION_CLASSES:
        raise ValueError("F1 validation must contain exactly Cars classes 49 through 81")
    train_counts = torch.bincount(train_labels.to(torch.int64), minlength=82)[:49]
    validation_counts = torch.bincount(validation_labels.to(torch.int64), minlength=82)[49:82]
    if bool((train_counts < 2).any()) or bool((validation_counts < 2).any()):
        raise ValueError("every F1 class must contain at least two examples")


def cross_label_token_permutation(labels: torch.Tensor, *, seed: int) -> torch.Tensor:
    """Return a deterministic bijection that gives every image another class's tokens.

    The construction starts from a valid class-block derangement and applies a fixed
    seeded sequence of validity-preserving individual swaps. It is fixed before
    optimization and never consults validation examples.
    """

    classes = _observed_classes(labels, name="shuffle")
    if len(classes) < 2:
        raise ValueError("a cross-label derangement requires at least two classes")
    labels_cpu = labels.to(device="cpu", dtype=torch.int64)
    grouped = torch.cat(
        [torch.nonzero(labels_cpu == label, as_tuple=False).flatten() for label in sorted(classes)]
    )
    maximum_class = max(int((labels_cpu == label).sum()) for label in classes)
    if maximum_class * 2 > labels_cpu.numel():
        raise ValueError("a cross-label derangement is impossible for a majority class")
    sources = torch.roll(grouped, shifts=-maximum_class)
    permutation = torch.empty_like(grouped)
    permutation[grouped] = sources
    generator = torch.Generator(device="cpu").manual_seed(seed)
    proposals = torch.randint(
        labels_cpu.numel(),
        (64 * labels_cpu.numel(), 2),
        generator=generator,
    )
    for left_tensor, right_tensor in proposals:
        left = int(left_tensor)
        right = int(right_tensor)
        if left == right:
            continue
        left_source = int(permutation[left])
        right_source = int(permutation[right])
        swap_is_valid = (
            labels_cpu[right_source] != labels_cpu[left]
            and labels_cpu[left_source] != labels_cpu[right]
        )
        if swap_is_valid:
            permutation[left], permutation[right] = right_source, left_source
    if bool((labels_cpu[permutation] == labels_cpu).any()):
        raise RuntimeError("cross-label derangement construction failed")
    for label in classes:
        target = labels_cpu == label
        required_sources = min(8, len(classes) - 1, int(target.sum()))
        if int(torch.unique(labels_cpu[permutation[target]]).numel()) < required_sources:
            raise RuntimeError("cross-label derangement did not mix enough source classes")
    return permutation.to(device=labels.device)


def leave_one_out_recall_at_one(
    global_embeddings: torch.Tensor,
    token_embeddings: torch.Tensor,
    token_weights: torch.Tensor,
    labels: torch.Tensor,
    *,
    set_weight: float,
    query_block: int,
) -> float:
    """Evaluate exact single-stage hybrid R@1 with deterministic lowest-index ties."""

    if not 0.0 <= set_weight <= 1.0 or query_block < 1:
        raise ValueError("retrieval weight or query block is invalid")
    count = global_embeddings.shape[0]
    if (
        global_embeddings.ndim != 2
        or token_embeddings.ndim != 3
        or token_weights.shape != token_embeddings.shape[:2]
        or token_embeddings.shape[0] != count
        or labels.shape != (count,)
        or count < 2
    ):
        raise ValueError("retrieval tensors have incompatible shapes")
    devices = {
        global_embeddings.device,
        token_embeddings.device,
        token_weights.device,
        labels.device,
    }
    if len(devices) != 1:
        raise ValueError("retrieval tensors must share a device")
    global_embeddings = F.normalize(global_embeddings.detach(), dim=-1)
    token_embeddings = F.normalize(token_embeddings.detach(), dim=-1)
    token_weights = token_weights.detach()
    correct = 0
    for start in range(0, count, query_block):
        stop = min(start + query_block, count)
        pooled = global_embeddings[start:stop].float() @ global_embeddings.float().T
        set_scores = fused_set_maxsim(
            token_embeddings[start:stop],
            token_embeddings,
            query_weights=token_weights[start:stop],
            gallery_weights=token_weights,
        )
        scores = (1.0 - set_weight) * pooled + set_weight * set_scores
        rows = torch.arange(stop - start, device=labels.device)
        columns = torch.arange(start, stop, device=labels.device)
        scores[rows, columns] = -torch.inf
        correct += int((labels[scores.argmax(dim=1)] == labels[start:stop]).sum())
    return correct / count
