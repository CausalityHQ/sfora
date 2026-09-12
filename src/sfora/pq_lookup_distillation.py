"""Label-free query-dependent lookup-table distillation for product codes."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

LOOKUP_MAXIMUM_GRADIENT_NORM = 1.0
LOOKUP_WEIGHT_DECAY = 0.01


@dataclass(frozen=True, slots=True)
class LookupTableDistillationSpec:
    """Shape authority for one additive correction to product-code lookup tables."""

    query_dimensions: int
    blocks: int
    codebook_size: int
    hidden_dimensions: int
    correction_dimensions: int

    def __post_init__(self) -> None:
        if (
            any(
                type(value) is not int or value < 1
                for value in (
                    self.query_dimensions,
                    self.blocks,
                    self.codebook_size,
                    self.hidden_dimensions,
                    self.correction_dimensions,
                )
            )
            or self.codebook_size > 256
        ):
            raise ValueError("lookup distillation specification differs")


class QueryDependentLookupScorer(nn.Module):
    """Correct product-code scores through query-dependent additive tables."""

    def __init__(self, spec: LookupTableDistillationSpec) -> None:
        super().__init__()
        if type(spec) is not LookupTableDistillationSpec:
            raise ValueError("lookup distillation specification differs")
        self.spec = spec
        self.query_network = nn.Sequential(
            nn.Linear(spec.query_dimensions, spec.hidden_dimensions),
            nn.GELU(),
            nn.Linear(spec.hidden_dimensions, spec.correction_dimensions),
        )
        self.code_embeddings = nn.Parameter(
            torch.zeros(
                (spec.blocks, spec.codebook_size, spec.correction_dimensions),
                dtype=torch.float32,
            )
        )

    def _validate_queries(self, queries: torch.Tensor) -> None:
        if (
            type(queries) is not torch.Tensor
            or queries.dtype != torch.float32
            or queries.ndim != 2
            or queries.shape[0] < 1
            or queries.shape[1] != self.spec.query_dimensions
            or queries.device != self.code_embeddings.device
            or not bool(torch.isfinite(queries).all())
        ):
            raise ValueError("lookup scorer input authority differs")

    def correction_tables(self, queries: torch.Tensor) -> torch.Tensor:
        """Build zero-offset correction tables once for each query."""

        self._validate_queries(queries)
        centered_embeddings = self.code_embeddings - self.code_embeddings.mean(dim=1, keepdim=True)
        query_features = self.query_network(queries)
        tables = torch.einsum("bd,mkd->bmk", query_features, centered_embeddings)
        return tables.contiguous()

    def forward(
        self,
        queries: torch.Tensor,
        candidate_codes: torch.Tensor,
        baseline_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Add the query table correction using one lookup per code byte."""

        self._validate_queries(queries)
        if (
            type(candidate_codes) is not torch.Tensor
            or candidate_codes.dtype != torch.uint8
            or candidate_codes.ndim != 3
            or candidate_codes.shape[0] != queries.shape[0]
            or candidate_codes.shape[1] < 2
            or candidate_codes.shape[2] != self.spec.blocks
            or candidate_codes.device != queries.device
            or bool((candidate_codes.to(torch.int16) >= self.spec.codebook_size).any())
            or type(baseline_scores) is not torch.Tensor
            or baseline_scores.dtype != torch.float32
            or baseline_scores.shape != candidate_codes.shape[:2]
            or baseline_scores.device != queries.device
            or not bool(torch.isfinite(baseline_scores).all())
        ):
            raise ValueError("lookup scorer input authority differs")
        tables = self.correction_tables(queries)
        correction = torch.zeros_like(baseline_scores)
        for block in range(self.spec.blocks):
            correction = correction + torch.gather(
                tables[:, block, :], 1, candidate_codes[:, :, block].long()
            )
        return (baseline_scores + correction).contiguous()

    def score_gallery(
        self,
        queries: torch.Tensor,
        gallery_codes: torch.Tensor,
        baseline_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Score one shared gallery without repeating its code matrix per query."""

        self._validate_queries(queries)
        if (
            type(gallery_codes) is not torch.Tensor
            or gallery_codes.dtype != torch.uint8
            or gallery_codes.ndim != 2
            or gallery_codes.shape[0] < 2
            or gallery_codes.shape[1] != self.spec.blocks
            or gallery_codes.device != queries.device
            or bool((gallery_codes.to(torch.int16) >= self.spec.codebook_size).any())
            or type(baseline_scores) is not torch.Tensor
            or baseline_scores.dtype != torch.float32
            or baseline_scores.shape != (len(queries), len(gallery_codes))
            or baseline_scores.device != queries.device
            or not bool(torch.isfinite(baseline_scores).all())
        ):
            raise ValueError("lookup scorer input authority differs")
        tables = self.correction_tables(queries)
        correction = torch.zeros_like(baseline_scores)
        for block in range(self.spec.blocks):
            correction = correction + tables[:, block, :][:, gallery_codes[:, block].long()]
        return (baseline_scores + correction).contiguous()


@dataclass(frozen=True, slots=True)
class LookupTableDistillationLoss:
    """Observable components of lookup-table ordering distillation."""

    total: torch.Tensor
    listwise_kl: torch.Tensor
    pairwise_bce: torch.Tensor
    score_mse: torch.Tensor


def lookup_table_distillation_loss(
    student_scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    pairs: torch.Tensor,
    *,
    temperature: float,
    listwise_weight: float,
    pairwise_weight: float,
    score_weight: float,
) -> LookupTableDistillationLoss:
    """Match listwise probabilities, contested pairs, and centered teacher scores."""

    if (
        type(student_scores) is not torch.Tensor
        or student_scores.dtype != torch.float32
        or student_scores.ndim != 2
        or student_scores.shape[0] < 1
        or student_scores.shape[1] < 2
        or type(teacher_scores) is not torch.Tensor
        or teacher_scores.dtype != torch.float32
        or teacher_scores.shape != student_scores.shape
        or teacher_scores.device != student_scores.device
        or type(pairs) is not torch.Tensor
        or pairs.dtype != torch.int64
        or pairs.ndim != 3
        or pairs.shape[0] != student_scores.shape[0]
        or pairs.shape[1] < 1
        or pairs.shape[2] != 2
        or pairs.device != student_scores.device
        or bool((pairs < 0).any())
        or bool((pairs >= student_scores.shape[1]).any())
        or bool((pairs[:, :, 0] == pairs[:, :, 1]).any())
        or not bool(torch.isfinite(student_scores).all())
        or not bool(torch.isfinite(teacher_scores).all())
        or type(temperature) is not float
        or not math.isfinite(temperature)
        or temperature <= 0
        or type(listwise_weight) is not float
        or not math.isfinite(listwise_weight)
        or listwise_weight < 0
        or type(pairwise_weight) is not float
        or not math.isfinite(pairwise_weight)
        or pairwise_weight < 0
        or type(score_weight) is not float
        or not math.isfinite(score_weight)
        or score_weight < 0
    ):
        raise ValueError("lookup-table distillation authority differs")
    detached_teacher = teacher_scores.detach()
    probabilities = F.softmax(detached_teacher / temperature, dim=-1)
    listwise_kl = F.kl_div(
        F.log_softmax(student_scores / temperature, dim=-1),
        probabilities,
        reduction="batchmean",
    )
    left = pairs[:, :, 0]
    right = pairs[:, :, 1]
    student_delta = student_scores.gather(1, left) - student_scores.gather(1, right)
    teacher_delta = detached_teacher.gather(1, left) - detached_teacher.gather(1, right)
    pairwise_bce = F.binary_cross_entropy_with_logits(
        student_delta / temperature, torch.sigmoid(teacher_delta / temperature)
    )
    centered_student = student_scores - student_scores.mean(dim=-1, keepdim=True)
    centered_teacher = detached_teacher - detached_teacher.mean(dim=-1, keepdim=True)
    score_mse = F.mse_loss(centered_student, centered_teacher)
    total = (
        listwise_weight * listwise_kl + pairwise_weight * pairwise_bce + score_weight * score_mse
    )
    if not bool(torch.isfinite(total)):
        raise RuntimeError("lookup-table distillation loss is nonfinite")
    return LookupTableDistillationLoss(total, listwise_kl, pairwise_bce, score_mse)


def fit_query_dependent_lookup_scorer(
    queries: torch.Tensor,
    candidate_codes: torch.Tensor,
    baseline_scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    pairs: torch.Tensor,
    *,
    spec: LookupTableDistillationSpec,
    seed: int,
    updates: int,
    batch_size: int,
    learning_rate: float,
    temperature: float,
    listwise_weight: float,
    pairwise_weight: float,
    score_weight: float,
    refresh_interval: int | None = None,
    refresh: Callable[
        [QueryDependentLookupScorer, int],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]
    | None = None,
) -> QueryDependentLookupScorer:
    """Fit one deterministic correction using only float-teacher ordering targets."""

    if (
        type(seed) is not int
        or seed < 0
        or type(updates) is not int
        or updates < 1
        or type(batch_size) is not int
        or batch_size < 1
        or batch_size > len(queries)
        or type(learning_rate) is not float
        or not math.isfinite(learning_rate)
        or learning_rate <= 0
        or (refresh_interval is None) != (refresh is None)
        or (
            refresh_interval is not None
            and (type(refresh_interval) is not int or refresh_interval < 1)
        )
    ):
        raise ValueError("lookup scorer fit authority differs")
    fork_devices = []
    if queries.device.type == "cuda":
        fork_devices = [
            queries.device.index
            if queries.device.index is not None
            else torch.cuda.current_device()
        ]
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(seed)
        model = QueryDependentLookupScorer(spec).to(queries.device)
        model(queries[:1], candidate_codes[:1], baseline_scores[:1])
        lookup_table_distillation_loss(
            baseline_scores,
            teacher_scores,
            pairs,
            temperature=temperature,
            listwise_weight=listwise_weight,
            pairwise_weight=pairwise_weight,
            score_weight=score_weight,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=LOOKUP_WEIGHT_DECAY
        )
        generator = torch.Generator().manual_seed(seed)
        schedule = torch.randperm(len(queries), generator=generator)
        offset = 0
        model.train()
        for step in range(updates):
            if refresh_interval is not None and step > 0 and step % refresh_interval == 0:
                assert refresh is not None
                candidate_codes, baseline_scores, teacher_scores, pairs = refresh(
                    model.eval(), step // refresh_interval
                )
                model.train()
                model(queries[:1], candidate_codes[:1], baseline_scores[:1])
                lookup_table_distillation_loss(
                    baseline_scores,
                    teacher_scores,
                    pairs,
                    temperature=temperature,
                    listwise_weight=listwise_weight,
                    pairwise_weight=pairwise_weight,
                    score_weight=score_weight,
                )
            if offset == len(schedule):
                schedule = torch.randperm(len(queries), generator=generator)
                offset = 0
            stop = min(offset + batch_size, len(schedule))
            indexes = schedule[offset:stop].to(queries.device)
            offset = stop
            optimizer.zero_grad(set_to_none=True)
            scores = model(queries[indexes], candidate_codes[indexes], baseline_scores[indexes])
            loss = lookup_table_distillation_loss(
                scores,
                teacher_scores[indexes],
                pairs[indexes],
                temperature=temperature,
                listwise_weight=listwise_weight,
                pairwise_weight=pairwise_weight,
                score_weight=score_weight,
            )
            loss.total.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(model.parameters(), LOOKUP_MAXIMUM_GRADIENT_NORM)
            optimizer.step()
        return model.eval()
