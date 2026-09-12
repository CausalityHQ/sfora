"""Uncertainty-aware candidate-set reranking for fixed product codes."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F

from sfora.product_quantization import ProductQuantizer


def aligned_product_squared_distances(
    queries: torch.Tensor,
    candidate_codes: torch.Tensor,
    quantizer: ProductQuantizer,
) -> torch.Tensor:
    """Score per-query candidate codes with the deployment ADC operation order."""

    if type(quantizer) is not ProductQuantizer:
        raise ValueError("aligned product distance authority differs")
    quantizer._validate_values(queries)
    if (
        type(candidate_codes) is not torch.Tensor
        or candidate_codes.dtype != torch.uint8
        or candidate_codes.ndim != 3
        or candidate_codes.shape[0] != len(queries)
        or candidate_codes.shape[1] < 1
        or candidate_codes.shape[2] != quantizer.spec.bytes_per_vector
        or candidate_codes.device != queries.device
        or bool((candidate_codes.to(torch.int16) >= quantizer.spec.codebook_size).any())
    ):
        raise ValueError("aligned product distance authority differs")
    distances = torch.zeros(
        (len(queries), candidate_codes.shape[1]),
        dtype=torch.float32,
        device=queries.device,
    )
    start = 0
    for block_index, (width, codebook) in enumerate(
        zip(quantizer.spec.block_dimensions, quantizer.codebooks, strict=True)
    ):
        query_block = queries[:, start : start + width]
        table = (query_block[:, None, :] - codebook[None, :, :]).square().sum(dim=-1)
        distances = distances + torch.gather(table, 1, candidate_codes[:, :, block_index].long())
        start += width
    return distances.contiguous()


def exact_product_shortlists(
    queries: torch.Tensor,
    gallery_codes: torch.Tensor,
    quantizer: ProductQuantizer,
    *,
    width: int,
    query_block_size: int,
) -> torch.Tensor:
    """Return exact leave-self-out ADC candidates with stable row-index ties."""

    if (
        type(quantizer) is not ProductQuantizer
        or type(width) is not int
        or width < 1
        or type(query_block_size) is not int
        or query_block_size < 1
    ):
        raise ValueError("product shortlist authority differs")
    quantizer._validate_values(queries)
    quantizer._validate_codes(gallery_codes)
    if queries.shape[0] != gallery_codes.shape[0] or width >= len(queries):
        raise ValueError("product shortlist authority differs")
    shortlists = []
    with torch.inference_mode():
        for start in range(0, len(queries), query_block_size):
            stop = min(start + query_block_size, len(queries))
            distances = quantizer.asymmetric_squared_distances(queries[start:stop], gallery_codes)
            distances[
                torch.arange(stop - start, device=queries.device),
                torch.arange(start, stop, device=queries.device),
            ] = torch.inf
            retained = width + 1
            values, indexes = torch.topk(distances, k=retained, dim=1, largest=False, sorted=False)
            ordinal_order = torch.argsort(indexes, dim=1, stable=True)
            indexes = indexes.gather(1, ordinal_order)
            values = values.gather(1, ordinal_order)
            distance_order = torch.argsort(values, dim=1, stable=True)
            indexes = indexes.gather(1, distance_order)
            values = values.gather(1, distance_order)
            ambiguous = values[:, width - 1] == values[:, width]
            for row in torch.nonzero(ambiguous, as_tuple=False).flatten().tolist():
                boundary = values[row, width - 1]
                candidates = torch.nonzero(distances[row] <= boundary, as_tuple=False).flatten()
                candidates = torch.sort(candidates).values
                candidate_values = distances[row, candidates]
                order = torch.argsort(candidate_values, stable=True)
                indexes[row, :width] = candidates[order[:width]]
            shortlists.append(indexes[:, :width])
    return torch.cat(shortlists, dim=0).contiguous()


@dataclass(frozen=True, slots=True)
class ProductCodeCandidateFeatures:
    """Exact query/code features for one bounded candidate set."""

    partial_dot_scores: torch.Tensor
    directional_variances: torch.Tensor
    code_residual_energy: torch.Tensor
    decoded_norm: torch.Tensor
    pairwise_code_similarity: torch.Tensor

    def concatenated(self) -> torch.Tensor:
        """Return per-candidate scalar features in their registered order."""

        return torch.cat(
            (
                self.partial_dot_scores,
                self.directional_variances,
                self.code_residual_energy,
                self.decoded_norm,
            ),
            dim=-1,
        ).contiguous()


class ProductCodeResidualStatistics(nn.Module):
    """Query-independent residual second moments for each PQ codeword."""

    def __init__(
        self,
        quantizer: ProductQuantizer,
        covariances: tuple[torch.Tensor, ...],
        residual_energies: tuple[torch.Tensor, ...],
    ) -> None:
        super().__init__()
        if (
            type(quantizer) is not ProductQuantizer
            or len(covariances) != quantizer.spec.bytes_per_vector
            or len(residual_energies) != quantizer.spec.bytes_per_vector
        ):
            raise ValueError("product-code residual statistics authority differs")
        self.spec = quantizer.spec
        for index, (width, codebook, covariance, energy) in enumerate(
            zip(
                self.spec.block_dimensions,
                quantizer.codebooks,
                covariances,
                residual_energies,
                strict=True,
            )
        ):
            if (
                type(covariance) is not torch.Tensor
                or covariance.dtype != torch.float32
                or covariance.shape != (self.spec.codebook_size, width, width)
                or type(energy) is not torch.Tensor
                or energy.dtype != torch.float32
                or energy.shape != (self.spec.codebook_size,)
                or not bool(torch.isfinite(covariance).all())
                or not bool(torch.isfinite(energy).all())
                or bool((energy < 0).any())
                or covariance.device != codebook.device
                or energy.device != codebook.device
            ):
                raise ValueError("product-code residual statistics authority differs")
            self.register_buffer(f"codebook_{index}", codebook.detach().clone().contiguous())
            self.register_buffer(f"covariance_{index}", covariance.detach().clone().contiguous())
            self.register_buffer(f"residual_energy_{index}", energy.detach().clone().contiguous())

    @classmethod
    def fit(
        cls, quantizer: ProductQuantizer, values: torch.Tensor, codes: torch.Tensor
    ) -> ProductCodeResidualStatistics:
        """Fit population residual second moments from query-independent rows."""

        quantizer._validate_values(values)
        quantizer._validate_codes(codes)
        if values.shape[0] != codes.shape[0]:
            raise ValueError("product-code residual statistics fit authority differs")
        decoded = quantizer.hard_decode(codes).detach()
        residuals = values.detach() - decoded
        covariances = []
        energies = []
        start = 0
        for block_index, width in enumerate(quantizer.spec.block_dimensions):
            block = residuals[:, start : start + width]
            block_codes = codes[:, block_index].long()
            covariance = torch.zeros(
                (quantizer.spec.codebook_size, width, width),
                dtype=torch.float32,
                device=values.device,
            )
            for codeword in range(quantizer.spec.codebook_size):
                selected = block[block_codes == codeword]
                if len(selected):
                    selected64 = selected.to(torch.float64)
                    covariance[codeword] = (selected64.T @ selected64 / len(selected)).to(
                        torch.float32
                    )
            covariances.append(covariance)
            energies.append(
                covariance.diagonal(dim1=-2, dim2=-1).sum(dim=-1).clamp_min(0).contiguous()
            )
            start += width
        return cls(quantizer, tuple(covariances), tuple(energies))

    def _validate_inputs(self, queries: torch.Tensor, candidate_codes: torch.Tensor) -> None:
        first = self.codebook_0
        if (
            type(queries) is not torch.Tensor
            or queries.dtype != torch.float32
            or queries.ndim != 2
            or queries.shape[0] < 1
            or queries.shape[1] != self.spec.dimensions
            or queries.device != first.device
            or type(candidate_codes) is not torch.Tensor
            or candidate_codes.dtype != torch.uint8
            or candidate_codes.ndim != 3
            or candidate_codes.shape[0] != queries.shape[0]
            or candidate_codes.shape[1] < 1
            or candidate_codes.shape[2] != self.spec.bytes_per_vector
            or candidate_codes.device != queries.device
            or bool((candidate_codes.to(torch.int16) >= self.spec.codebook_size).any())
            or not bool(torch.isfinite(queries).all())
        ):
            raise ValueError("product-code candidate feature authority differs")

    def candidate_features(
        self, queries: torch.Tensor, candidate_codes: torch.Tensor
    ) -> ProductCodeCandidateFeatures:
        """Derive uncertainty and set-context features without database-side bytes."""

        self._validate_inputs(queries, candidate_codes)
        partials = []
        variances = []
        energies = []
        decoded_blocks = []
        start = 0
        for index, width in enumerate(self.spec.block_dimensions):
            query_block = queries[:, start : start + width]
            indexes = candidate_codes[:, :, index].long()
            codebook = getattr(self, f"codebook_{index}")
            covariance = getattr(self, f"covariance_{index}")
            energy = getattr(self, f"residual_energy_{index}")
            selected = codebook[indexes]
            decoded_blocks.append(selected)
            partials.append(torch.einsum("bd,bcd->bc", query_block, selected))
            selected_covariance = covariance[indexes]
            variances.append(
                torch.einsum("bd,bcdk,bk->bc", query_block, selected_covariance, query_block)
            )
            energies.append(energy[indexes])
            start += width
        decoded = torch.cat(decoded_blocks, dim=-1)
        raw_pairwise = torch.matmul(decoded, decoded.transpose(1, 2))
        return ProductCodeCandidateFeatures(
            partial_dot_scores=torch.stack(partials, dim=-1).contiguous(),
            directional_variances=torch.stack(variances, dim=-1).clamp_min(0).contiguous(),
            code_residual_energy=torch.stack(energies, dim=-1).contiguous(),
            decoded_norm=torch.linalg.vector_norm(decoded, dim=-1, keepdim=True).contiguous(),
            pairwise_code_similarity=(
                0.5 * (raw_pairwise + raw_pairwise.transpose(1, 2))
            ).contiguous(),
        )


class ProductCodeResidualDecoder(nn.Module):
    """Decode systematic cross-block residual structure without changing product codes."""

    def __init__(self, quantizer: ProductQuantizer, *, hidden_dimensions: int) -> None:
        super().__init__()
        if (
            type(quantizer) is not ProductQuantizer
            or type(hidden_dimensions) is not int
            or hidden_dimensions < 1
        ):
            raise ValueError("product-code residual decoder authority differs")
        self.spec = quantizer.spec
        output = nn.Linear(hidden_dimensions, self.spec.dimensions)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        self.network = nn.Sequential(
            nn.Linear(self.spec.dimensions, hidden_dimensions), nn.GELU(), output
        )
        self.to(quantizer.codebooks[0].device)

    def forward(self, decoded: torch.Tensor) -> torch.Tensor:
        """Return the decoded vectors plus their learned query-independent residuals."""

        if (
            type(decoded) is not torch.Tensor
            or decoded.dtype != torch.float32
            or decoded.ndim != 2
            or decoded.shape[0] < 1
            or decoded.shape[1] != self.spec.dimensions
            or decoded.device != self.network[0].weight.device
            or not bool(torch.isfinite(decoded).all())
        ):
            raise ValueError("product-code residual decoder input differs")
        return cast(torch.Tensor, (decoded + self.network(decoded)).contiguous())

    def decode(
        self, codes: torch.Tensor, quantizer: ProductQuantizer
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode codes and shorten each block correction to its selected Voronoi cell."""

        if type(quantizer) is not ProductQuantizer or quantizer.spec != self.spec:
            raise ValueError("product-code residual decoder codec differs")
        quantizer._validate_codes(codes)
        if codes.device != self.network[0].weight.device:
            raise ValueError("product-code residual decoder codec differs")
        decoded = quantizer.hard_decode(codes)
        proposed = self(decoded)
        corrected_blocks = []
        steps = []
        start = 0
        row_indexes = torch.arange(len(codes), device=codes.device)
        for block_index, (width, codebook) in enumerate(
            zip(self.spec.block_dimensions, quantizer.codebooks, strict=True)
        ):
            current = decoded[:, start : start + width]
            delta = proposed[:, start : start + width] - current
            differences = current[:, None, :] - codebook[None, :, :]
            numerator = differences.square().sum(dim=-1)
            denominator = -2.0 * torch.einsum("bd,bkd->bk", delta, differences)
            bounds = torch.where(
                denominator > 0,
                numerator / denominator.clamp_min(torch.finfo(torch.float32).tiny),
                torch.inf,
            )
            bounds[row_indexes, codes[:, block_index].long()] = torch.inf
            step = bounds.amin(dim=1).clamp(0.0, 1.0)
            corrected_blocks.append(current + step[:, None] * delta)
            steps.append(step)
            start += width
        corrected = torch.cat(corrected_blocks, dim=1).contiguous()
        step_values = torch.stack(steps, dim=1).contiguous()
        if not bool(torch.isfinite(corrected).all()) or not bool(torch.isfinite(step_values).all()):
            raise RuntimeError("product-code residual decoder output is nonfinite")
        return corrected, step_values

    def candidate_pairwise(
        self, candidate_codes: torch.Tensor, quantizer: ProductQuantizer
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cosine geometry and cell steps for bounded candidate code sets."""

        if (
            type(candidate_codes) is not torch.Tensor
            or candidate_codes.dtype != torch.uint8
            or candidate_codes.ndim != 3
            or candidate_codes.shape[0] < 1
            or candidate_codes.shape[1] < 1
            or candidate_codes.shape[2] != self.spec.bytes_per_vector
        ):
            raise ValueError("product-code residual decoder candidates differ")
        batch, candidates, blocks = candidate_codes.shape
        flat_codes = candidate_codes.reshape(batch * candidates, blocks).contiguous()
        corrected, flat_steps = self.decode(flat_codes, quantizer)
        normalized = F.normalize(corrected, dim=1).reshape(batch, candidates, -1)
        raw_pairwise = torch.matmul(normalized, normalized.transpose(1, 2))
        pairwise = (0.5 * (raw_pairwise + raw_pairwise.transpose(1, 2))).contiguous()
        return pairwise, flat_steps.reshape(batch, candidates, blocks).contiguous()


def fit_product_code_residual_decoder(
    quantizer: ProductQuantizer,
    values: torch.Tensor,
    codes: torch.Tensor,
    *,
    hidden_dimensions: int,
    seed: int,
    updates: int,
    batch_size: int,
    learning_rate: float,
) -> ProductCodeResidualDecoder:
    """Fit a deterministic pointwise residual decoder for fixed product codes."""

    if (
        type(quantizer) is not ProductQuantizer
        or type(seed) is not int
        or seed < 0
        or type(updates) is not int
        or updates < 1
        or type(batch_size) is not int
        or batch_size < 1
        or batch_size > len(values)
        or type(learning_rate) is not float
        or not math.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError("product-code residual decoder fit authority differs")
    quantizer._validate_values(values)
    quantizer._validate_codes(codes)
    if len(values) != len(codes):
        raise ValueError("product-code residual decoder fit authority differs")
    fork_devices = []
    if values.device.type == "cuda":
        fork_devices = [
            values.device.index if values.device.index is not None else torch.cuda.current_device()
        ]
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(seed)
        decoder = ProductCodeResidualDecoder(quantizer, hidden_dimensions=hidden_dimensions)
        optimizer = torch.optim.AdamW(decoder.parameters(), lr=learning_rate, weight_decay=0.01)
        generator = torch.Generator().manual_seed(seed)
        schedule = torch.randperm(len(values), generator=generator)
        offset = 0
        decoded = quantizer.hard_decode(codes).detach()
        decoder.train()
        for _ in range(updates):
            if offset + batch_size > len(schedule):
                schedule = torch.randperm(len(values), generator=generator)
                offset = 0
            indexes = schedule[offset : offset + batch_size].to(values.device)
            offset += batch_size
            optimizer.zero_grad(set_to_none=True)
            predicted = decoder(decoded[indexes])
            pointwise = F.mse_loss(predicted, values[indexes])
            directional = 1.0 - F.cosine_similarity(predicted, values[indexes], dim=1).mean()
            loss = pointwise + 0.1 * directional
            loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            optimizer.step()
        return decoder.eval()


class _CandidateAttentionBlock(nn.Module):
    def __init__(self, dimensions: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.head_dimensions = dimensions // heads
        self.norm1 = nn.LayerNorm(dimensions)
        self.qkv = nn.Linear(dimensions, 3 * dimensions)
        self.pairwise_scale = nn.Parameter(torch.zeros(heads))
        self.output = nn.Linear(dimensions, dimensions)
        self.norm2 = nn.LayerNorm(dimensions)
        self.feed_forward = nn.Sequential(
            nn.Linear(dimensions, 4 * dimensions),
            nn.GELU(),
            nn.Linear(4 * dimensions, dimensions),
        )

    def forward(self, values: torch.Tensor, pairwise: torch.Tensor) -> torch.Tensor:
        batch, candidates, dimensions = values.shape
        normalized = self.norm1(values)
        qkv = self.qkv(normalized).reshape(batch, candidates, 3, self.heads, self.head_dimensions)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        logits = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(self.head_dimensions)
        logits = logits + self.pairwise_scale.reshape(1, self.heads, 1, 1) * pairwise[:, None]
        attended = torch.matmul(torch.softmax(logits, dim=-1), value)
        attended = attended.transpose(1, 2).reshape(batch, candidates, dimensions)
        values = values + self.output(attended)
        return cast(torch.Tensor, values + self.feed_forward(self.norm2(values)))


class CandidateSetReranker(nn.Module):
    """Permutation-equivariant residual scorer for a bounded PQ candidate set."""

    def __init__(self, *, blocks: int, model_dimensions: int, heads: int, layers: int) -> None:
        super().__init__()
        if (
            type(blocks) is not int
            or blocks < 1
            or type(model_dimensions) is not int
            or model_dimensions < 1
            or type(heads) is not int
            or heads < 1
            or model_dimensions % heads
            or type(layers) is not int
            or layers < 1
        ):
            raise ValueError("candidate reranker architecture differs")
        self.blocks = blocks
        self.input = nn.Linear(3 * blocks + 1, model_dimensions)
        self.layers = nn.ModuleList(
            _CandidateAttentionBlock(model_dimensions, heads) for _ in range(layers)
        )
        self.output = nn.Linear(model_dimensions, 1)
        self.residual_scale = nn.Parameter(torch.zeros(()))

    def forward(
        self, baseline_scores: torch.Tensor, features: torch.Tensor, pairwise: torch.Tensor
    ) -> torch.Tensor:
        """Return baseline plus a learned set-conditioned residual score."""

        if (
            type(baseline_scores) is not torch.Tensor
            or baseline_scores.dtype != torch.float32
            or baseline_scores.ndim != 2
            or baseline_scores.shape[0] < 1
            or baseline_scores.shape[1] < 1
            or type(features) is not torch.Tensor
            or features.dtype != torch.float32
            or features.shape != (*baseline_scores.shape, 3 * self.blocks + 1)
            or type(pairwise) is not torch.Tensor
            or pairwise.dtype != torch.float32
            or pairwise.shape
            != (
                baseline_scores.shape[0],
                baseline_scores.shape[1],
                baseline_scores.shape[1],
            )
            or baseline_scores.device != features.device
            or pairwise.device != features.device
            or not bool(torch.isfinite(baseline_scores).all())
            or not bool(torch.isfinite(features).all())
            or not bool(torch.isfinite(pairwise).all())
            or not torch.equal(pairwise, pairwise.transpose(1, 2))
            or baseline_scores.device != self.input.weight.device
        ):
            raise ValueError("candidate reranker input authority differs")
        hidden = self.input(features)
        for layer in self.layers:
            hidden = layer(hidden, pairwise)
        correction = self.output(hidden).squeeze(-1)
        return cast(torch.Tensor, (baseline_scores + self.residual_scale * correction).contiguous())


def fit_candidate_set_reranker(
    baseline_scores: torch.Tensor,
    features: torch.Tensor,
    pairwise: torch.Tensor,
    teacher_scores: torch.Tensor,
    *,
    blocks: int,
    model_dimensions: int,
    heads: int,
    layers: int,
    seed: int,
    updates: int,
    batch_size: int,
    learning_rate: float,
    temperature: float,
    score_weight: float,
) -> CandidateSetReranker:
    """Fit one deterministic residual reranker from label-free teacher lists."""

    if (
        type(seed) is not int
        or seed < 0
        or type(updates) is not int
        or updates < 1
        or type(batch_size) is not int
        or batch_size < 1
        or batch_size > len(baseline_scores)
        or type(learning_rate) is not float
        or not math.isfinite(learning_rate)
        or learning_rate <= 0
        or type(teacher_scores) is not torch.Tensor
        or teacher_scores.dtype != torch.float32
        or teacher_scores.shape != baseline_scores.shape
        or teacher_scores.device != baseline_scores.device
        or not bool(torch.isfinite(teacher_scores).all())
    ):
        raise ValueError("candidate reranker fit authority differs")
    fork_devices = []
    if baseline_scores.device.type == "cuda":
        fork_devices = [
            baseline_scores.device.index
            if baseline_scores.device.index is not None
            else torch.cuda.current_device()
        ]
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(seed)
        model = CandidateSetReranker(
            blocks=blocks,
            model_dimensions=model_dimensions,
            heads=heads,
            layers=layers,
        ).to(baseline_scores.device)
        model(baseline_scores[:1], features[:1], pairwise[:1])
        ordinary = [
            parameter for name, parameter in model.named_parameters() if name != "residual_scale"
        ]
        optimizer = torch.optim.AdamW(
            (
                {"params": ordinary, "lr": learning_rate},
                {"params": [model.residual_scale], "lr": 10.0 * learning_rate},
            ),
            weight_decay=0.01,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(step, updates) / updates)),
        )
        generator = torch.Generator().manual_seed(seed)
        schedule = torch.randperm(len(baseline_scores), generator=generator)
        offset = 0
        model.train()
        for _ in range(updates):
            if offset + batch_size > len(schedule):
                schedule = torch.randperm(len(baseline_scores), generator=generator)
                offset = 0
            indexes = schedule[offset : offset + batch_size].to(baseline_scores.device)
            offset += batch_size
            optimizer.zero_grad(set_to_none=True)
            scores = model(baseline_scores[indexes], features[indexes], pairwise[indexes])
            loss = candidate_set_distillation_loss(
                scores,
                teacher_scores[indexes],
                temperature=temperature,
                score_weight=score_weight,
            )
            loss.total.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        return model.eval()


@dataclass(frozen=True, slots=True)
class CandidateSetDistillationLoss:
    """Observable components of listwise score distillation."""

    total: torch.Tensor
    listwise_kl: torch.Tensor
    score_mse: torch.Tensor


@dataclass(frozen=True, slots=True)
class CandidateSetSupervisedLoss:
    """Observable label-free and uniform-positive candidate refinement losses."""

    total: torch.Tensor
    distillation: torch.Tensor
    positive_listwise: torch.Tensor
    positive_coverage: float


def candidate_set_distillation_loss(
    student_scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    *,
    temperature: float,
    score_weight: float,
) -> CandidateSetDistillationLoss:
    """Distill float-query shortlist ordering into one candidate-set scorer."""

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
        or not bool(torch.isfinite(student_scores).all())
        or not bool(torch.isfinite(teacher_scores).all())
        or type(temperature) is not float
        or not math.isfinite(temperature)
        or temperature <= 0
        or type(score_weight) is not float
        or not math.isfinite(score_weight)
        or score_weight < 0
    ):
        raise ValueError("candidate-set distillation authority differs")
    probabilities = F.softmax(teacher_scores.detach() / temperature, dim=-1)
    listwise_kl = F.kl_div(
        F.log_softmax(student_scores / temperature, dim=-1),
        probabilities,
        reduction="batchmean",
    )
    centered_student = student_scores - student_scores.mean(dim=-1, keepdim=True)
    centered_teacher = teacher_scores.detach() - teacher_scores.detach().mean(dim=-1, keepdim=True)
    score_mse = F.mse_loss(centered_student, centered_teacher)
    total = listwise_kl + score_weight * score_mse
    if not bool(torch.isfinite(total)):
        raise RuntimeError("candidate-set distillation loss is nonfinite")
    return CandidateSetDistillationLoss(total, listwise_kl, score_mse)


def candidate_set_supervised_loss(
    student_scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    positive_mask: torch.Tensor,
    *,
    temperature: float,
    score_weight: float,
    supervised_weight: float,
) -> CandidateSetSupervisedLoss:
    """Combine float-score distillation with uniform-positive shortlist supervision."""

    if (
        type(positive_mask) is not torch.Tensor
        or positive_mask.dtype != torch.bool
        or positive_mask.shape != student_scores.shape
        or positive_mask.device != student_scores.device
        or type(supervised_weight) is not float
        or not math.isfinite(supervised_weight)
        or supervised_weight < 0
    ):
        raise ValueError("candidate-set supervised loss authority differs")
    distillation = candidate_set_distillation_loss(
        student_scores,
        teacher_scores,
        temperature=temperature,
        score_weight=score_weight,
    ).total
    valid = positive_mask.any(dim=1)
    if bool(valid.any()):
        targets = positive_mask[valid].to(student_scores.dtype)
        targets = targets / targets.sum(dim=1, keepdim=True)
        positive_listwise = (
            -(targets * F.log_softmax(student_scores[valid] / temperature, dim=1)).sum(dim=1).mean()
        )
    else:
        positive_listwise = student_scores.sum() * 0.0
    total = distillation + supervised_weight * positive_listwise
    if not bool(torch.isfinite(total)):
        raise RuntimeError("candidate-set supervised loss is nonfinite")
    return CandidateSetSupervisedLoss(
        total=total,
        distillation=distillation,
        positive_listwise=positive_listwise,
        positive_coverage=float(valid.to(torch.float32).mean()),
    )


def refine_supervised_candidate_set_reranker(
    source: CandidateSetReranker,
    baseline_scores: torch.Tensor,
    features: torch.Tensor,
    pairwise: torch.Tensor,
    teacher_scores: torch.Tensor,
    positive_mask: torch.Tensor,
    *,
    seed: int,
    updates: int,
    batch_size: int,
    learning_rate: float,
    temperature: float,
    score_weight: float,
    supervised_weight: float,
) -> CandidateSetReranker:
    """Refine a copied candidate reranker using fitting-set positive identities."""

    if (
        type(source) is not CandidateSetReranker
        or type(seed) is not int
        or seed < 0
        or type(updates) is not int
        or updates < 1
        or type(batch_size) is not int
        or batch_size < 1
        or batch_size > len(baseline_scores)
        or type(learning_rate) is not float
        or not math.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError("candidate-set supervised refinement authority differs")
    source(baseline_scores[:1], features[:1], pairwise[:1])
    candidate_set_supervised_loss(
        baseline_scores,
        teacher_scores,
        positive_mask,
        temperature=temperature,
        score_weight=score_weight,
        supervised_weight=supervised_weight,
    )
    fork_devices = []
    if baseline_scores.device.type == "cuda":
        fork_devices = [
            baseline_scores.device.index
            if baseline_scores.device.index is not None
            else torch.cuda.current_device()
        ]
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(seed)
        model = copy.deepcopy(source)
        optimizer = torch.optim.AdamW(model.parameters(), learning_rate, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=updates, eta_min=0.1 * learning_rate
        )
        generator = torch.Generator().manual_seed(seed)
        schedule = torch.randperm(len(baseline_scores), generator=generator)
        offset = 0
        model.train()
        for _ in range(updates):
            if offset + batch_size > len(schedule):
                schedule = torch.randperm(len(baseline_scores), generator=generator)
                offset = 0
            indexes = schedule[offset : offset + batch_size].to(baseline_scores.device)
            offset += batch_size
            optimizer.zero_grad(set_to_none=True)
            scores = model(baseline_scores[indexes], features[indexes], pairwise[indexes])
            loss = candidate_set_supervised_loss(
                scores,
                teacher_scores[indexes],
                positive_mask[indexes],
                temperature=temperature,
                score_weight=score_weight,
                supervised_weight=supervised_weight,
            )
            loss.total.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        return model.eval()
