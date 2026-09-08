import hashlib
import math
import struct
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest
import torch
from torch.nn import functional as F

import sfora.packed_int4 as packed_int4_module
from sfora.joint_relational_compaction import (
    JointRelationalEncoder,
    PackedInt8Embeddings,
    RelationalLinearEncoder,
    RelationalLinearTrainingConfig,
    fit_relational_linear_compaction,
    fit_relational_linear_encoder,
    fixed_int8_unit_codes,
    neighborhood_distribution_kl,
    pack_int8_unit_embeddings,
)
from sfora.packed_int4 import (
    PackedInt4Embeddings,
    ResidentInt4Gallery,
    fixed_int4_unit_codes,
    pack_int4_unit_embeddings,
)


def _unit(rows: int, dimensions: int) -> torch.Tensor:
    values = torch.arange(1, rows * dimensions + 1, dtype=torch.float32).reshape(rows, dimensions)
    return F.normalize(values, dim=1)


def test_module_import_does_not_depend_on_experimental_research_modules() -> None:
    code = """
import importlib.abc
import sys

class BlockExperimental(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "sfora.split_code_anchor":
            raise ModuleNotFoundError(fullname)
        return None

sys.meta_path.insert(0, BlockExperimental())
import sfora.joint_relational_compaction
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_encoder_starts_at_pca_and_residual_can_change_geometry() -> None:
    basis = torch.eye(4, dtype=torch.float32)[:2]
    encoder = JointRelationalEncoder(basis, hidden_dimensions=3, seed=17)
    values = _unit(3, 4)

    expected = F.normalize(values @ basis.T, dim=1)
    torch.testing.assert_close(encoder(values), expected)

    with torch.no_grad():
        encoder.residual_input.weight.fill_(0.25)
        encoder.residual_output.weight[0].fill_(0.25)
    assert not torch.equal(encoder(values), expected)


def test_linear_encoder_is_exactly_a_shared_normalized_projection() -> None:
    basis = torch.eye(4, dtype=torch.float32)[:2]
    encoder = RelationalLinearEncoder(basis)
    values = _unit(3, 4)

    torch.testing.assert_close(encoder(values), F.normalize(values @ basis.T, dim=1))
    assert sum(parameter.numel() for parameter in encoder.parameters()) == 8
    assert all(
        module.bias is None for module in encoder.modules() if isinstance(module, torch.nn.Linear)
    )


def test_linear_encoder_has_a_canonical_weight_artifact() -> None:
    encoder = RelationalLinearEncoder(torch.eye(4, dtype=torch.float32)[:2])

    wire = encoder.to_bytes()
    restored = RelationalLinearEncoder.from_bytes(wire)

    assert wire.startswith(b"SFORA-RL1")
    assert len(wire) == 9 + 8 + 2 * 4 * 4
    assert restored.to_bytes() == wire
    torch.testing.assert_close(restored.projection.weight, encoder.projection.weight)
    with pytest.raises(ValueError, match="encoder byte authority"):
        RelationalLinearEncoder.from_bytes(wire + b"trailing")


def test_committed_deployment_model_is_authenticated_and_replayable() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "docs/evidence/relational_linear_compaction/relational-linear-v24.sfora-rl1"
    )
    wire = path.read_bytes()

    assert hashlib.sha256(wire).hexdigest() == (
        "c46d5c7eff99b4962b9491688ac9a1ad5d345ea1e2bc9c4bd7ae3bfc0b186521"
    )
    model = RelationalLinearEncoder.from_bytes(wire)
    assert model.projection.weight.shape == (64, 768)
    assert model.to_bytes() == wire


def test_encoder_is_seeded_and_rejects_invalid_authority() -> None:
    basis = torch.eye(4, dtype=torch.float32)[:2]
    first = JointRelationalEncoder(basis, hidden_dimensions=3, seed=1729)
    second = JointRelationalEncoder(basis, hidden_dimensions=3, seed=1729)
    assert torch.equal(first.residual_input.weight, second.residual_input.weight)

    with pytest.raises(ValueError, match="basis authority"):
        JointRelationalEncoder(basis.double(), hidden_dimensions=3, seed=1729)
    with pytest.raises(ValueError, match="input authority"):
        first(torch.ones(2, 5))


def test_neighborhood_kl_accepts_distinct_embedding_widths_and_has_zero_identity() -> None:
    student = _unit(4, 3)
    teacher = student @ torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    )

    loss = neighborhood_distribution_kl(student, teacher, temperature=0.05)
    assert float(loss) == pytest.approx(0.0, abs=1e-7)

    changed = teacher.roll(1, dims=0)
    assert float(neighborhood_distribution_kl(student, changed, temperature=0.05)) > 0
    with pytest.raises(ValueError, match="relation authority"):
        neighborhood_distribution_kl(student[:1], teacher[:1], temperature=0.05)


def test_fixed_int8_codes_are_64_bytes_and_preserve_unit_geometry() -> None:
    values = _unit(5, 64)
    codes, restored = fixed_int8_unit_codes(values)

    assert codes.dtype == torch.int8
    assert codes.shape == (5, 64)
    assert codes.element_size() * codes.shape[1] == 64
    torch.testing.assert_close(
        torch.linalg.vector_norm(restored, dim=1), torch.ones(5), atol=1e-6, rtol=0
    )
    assert bool(torch.isfinite(restored).all())


def test_packed_int8_embeddings_are_66_bytes_and_round_trip_exactly() -> None:
    values = _unit(5, 64)
    packed = pack_int8_unit_embeddings(values)
    wire = packed.to_bytes()

    assert len(wire) == 5 * 66
    assert packed.bytes_per_vector == 66
    restored = PackedInt8Embeddings.from_bytes(wire, count=5, dimensions=64)
    assert torch.equal(restored.codes, packed.codes)
    assert torch.equal(restored.inverse_norms, packed.inverse_norms)
    torch.testing.assert_close(
        restored.restore(), fixed_int8_unit_codes(values)[1], atol=2e-4, rtol=2e-4
    )


def test_packed_int8_cosine_matches_restored_float_cosine() -> None:
    gallery = pack_int8_unit_embeddings(_unit(5, 64))
    queries = pack_int8_unit_embeddings(_unit(3, 64).roll(1, dims=1))

    expected = queries.restore() @ gallery.restore().T
    torch.testing.assert_close(queries.cosine_similarity(gallery), expected)

    with pytest.raises(ValueError, match="packed int8 byte authority"):
        PackedInt8Embeddings.from_bytes(b"short", count=1, dimensions=64)
    with pytest.raises(ValueError, match="packed int8 embedding authority"):
        PackedInt8Embeddings(
            codes=torch.zeros((1, 64), dtype=torch.int8),
            inverse_norms=torch.ones(1, dtype=torch.float16),
        )
    with pytest.raises(ValueError, match="packed int8 embedding authority"):
        PackedInt8Embeddings(
            codes=gallery.codes,
            inverse_norms=torch.ones(len(gallery.codes), dtype=torch.float16),
        )


def test_packed_int8_accepts_interoperable_one_ulp_inverse_norm() -> None:
    packed = pack_int8_unit_embeddings(_unit(4, 64))
    alternative = packed.inverse_norms.clone()
    alternative[0] = torch.nextafter(alternative[0], torch.tensor(torch.inf, dtype=torch.float16))

    value = PackedInt8Embeddings(codes=packed.codes, inverse_norms=alternative)

    assert torch.equal(value.inverse_norms, alternative)


def test_packed_int4_embeddings_are_66_bytes_at_128d_and_round_trip_exactly() -> None:
    values = _unit(5, 128)
    packed = pack_int4_unit_embeddings(values)
    wire = packed.to_bytes()

    assert packed.packed_codes.dtype == torch.uint8
    assert packed.packed_codes.shape == (5, 64)
    assert packed.dimensions == 128
    assert packed.bytes_per_vector == 66
    assert len(wire) == 5 * 66
    restored = PackedInt4Embeddings.from_bytes(wire, count=5, dimensions=128)
    assert torch.equal(restored.packed_codes, packed.packed_codes)
    assert torch.equal(restored.inverse_norms, packed.inverse_norms)
    assert torch.equal(restored.signed_codes(), packed.signed_codes())
    torch.testing.assert_close(restored.restore(), packed.restore())


def test_packed_int4_uses_canonical_low_then_high_twos_complement_nibbles() -> None:
    values = F.normalize(torch.tensor([[1.0, -1.0, 0.5, -0.5]], dtype=torch.float32), dim=1)

    codes, restored = fixed_int4_unit_codes(values)
    packed = pack_int4_unit_embeddings(values)

    assert codes.tolist() == [[7, -7, 4, -4]]
    assert packed.packed_codes.tolist() == [[0x97, 0xC4]]
    assert packed.to_bytes() == bytes((0x97, 0xC4)) + struct.pack("<e", 1.0 / math.sqrt(130))
    torch.testing.assert_close(packed.restore(), restored, atol=2e-4, rtol=2e-4)


def test_packed_int4_cosine_matches_restored_float_cosine() -> None:
    gallery = pack_int4_unit_embeddings(_unit(5, 128))
    queries = pack_int4_unit_embeddings(_unit(3, 128).roll(1, dims=1))

    expected = queries.restore() @ gallery.restore().T

    torch.testing.assert_close(queries.cosine_similarity(gallery), expected)


def test_packed_int4_float_query_similarity_is_exact_and_strict() -> None:
    gallery = pack_int4_unit_embeddings(_unit(5, 128))
    queries = _unit(3, 128).roll(1, dims=1).contiguous()

    expected = queries @ gallery.restore().T

    torch.testing.assert_close(gallery.float_query_similarity(queries), expected)
    with pytest.raises(ValueError, match="float query authority"):
        gallery.float_query_similarity(queries[:, :-2].contiguous())
    with pytest.raises(ValueError, match="float query authority"):
        gallery.float_query_similarity(queries * 2.0)
    with pytest.raises(ValueError, match="float query authority"):
        gallery.float_query_similarity(queries, device="cpu")  # type: ignore[arg-type]


@pytest.mark.skipif(
    not packed_int4_module._cpu_int_mm_available(),
    reason="exact CPU int8 matrix kernel unavailable",
)
def test_resident_int4_gallery_uses_integer_kernel_with_exact_score_and_rank_parity() -> None:
    generator = torch.Generator().manual_seed(17)
    gallery_values = F.normalize(torch.randn((17, 128), generator=generator), dim=1).contiguous()
    query_values = F.normalize(torch.randn((5, 128), generator=generator), dim=1).contiguous()
    gallery = pack_int4_unit_embeddings(gallery_values)
    queries = pack_int4_unit_embeddings(query_values)

    resident = ResidentInt4Gallery.from_packed(gallery)
    expected = queries.cosine_similarity(gallery)
    observed = resident.score_queries(queries=queries)

    assert resident.resident_bytes_per_vector == 130
    assert resident.gallery_codes_transposed.dtype == torch.int8
    assert resident.gallery_codes_transposed.shape == (128, 17)
    assert resident.gallery_codes_transposed.is_contiguous()
    assert torch.equal(observed, expected)
    assert torch.equal(
        torch.argsort(observed, dim=1, descending=True, stable=True),
        torch.argsort(expected, dim=1, descending=True, stable=True),
    )


def test_resident_int4_gallery_rejects_noncanonical_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gallery = pack_int4_unit_embeddings(_unit(5, 128))
    resident = ResidentInt4Gallery.from_packed(gallery)
    queries = pack_int4_unit_embeddings(_unit(2, 128))

    with pytest.raises(ValueError, match="resident int4 gallery authority"):
        ResidentInt4Gallery(
            gallery_codes_transposed=resident.gallery_codes_transposed.T,
            inverse_norms=resident.inverse_norms,
            dimensions=128,
        )
    strided = torch.ones((128, 10), dtype=torch.int8)[:, ::2]
    assert strided.shape == (128, 5) and not strided.is_contiguous()
    with pytest.raises(ValueError, match="resident int4 gallery authority"):
        ResidentInt4Gallery(
            gallery_codes_transposed=strided,
            inverse_norms=resident.inverse_norms,
            dimensions=128,
        )
    with pytest.raises(ValueError, match="resident int4 gallery authority"):
        ResidentInt4Gallery(
            gallery_codes_transposed=torch.full((128, 1), 8, dtype=torch.int8),
            inverse_norms=torch.tensor([1.0 / math.sqrt(128 * 64)], dtype=torch.float16),
            dimensions=128,
        )
    oversized = torch.ones((342_394, 1), dtype=torch.int8)
    with pytest.raises(ValueError, match="resident int4 gallery authority"):
        ResidentInt4Gallery(
            gallery_codes_transposed=oversized,
            inverse_norms=torch.tensor([1.0 / math.sqrt(342_394)], dtype=torch.float16),
            dimensions=342_394,
        )
    oversized_packed = PackedInt4Embeddings(
        packed_codes=torch.full((1, 171_197), 0x11, dtype=torch.uint8),
        inverse_norms=torch.tensor([1.0 / math.sqrt(342_394)], dtype=torch.float16),
        dimensions=342_394,
    )
    with pytest.raises(ValueError, match="packed int4 similarity authority"):
        oversized_packed.cosine_similarity(oversized_packed)
    with pytest.raises(ValueError, match="resident int4 similarity authority"):
        resident.score_queries(queries=pack_int4_unit_embeddings(_unit(2, 126)))
    monkeypatch.setattr(torch, "_int_mm", None)
    packed_int4_module._cpu_int_mm_available.cache_clear()
    with pytest.raises(RuntimeError, match="integer kernel is unavailable"):
        resident.score_queries(queries=queries)
    assert torch.equal(
        resident.score_queries(queries=queries, require_integer=False),
        queries.cosine_similarity(gallery),
    )
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        autocast_observed = resident.score_queries(queries=queries, require_integer=False)
    assert torch.equal(autocast_observed, queries.cosine_similarity(gallery))

    def unavailable_integer_mm(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        del left, right
        raise NotImplementedError("CPU kernel is not registered")

    monkeypatch.setattr(torch, "_int_mm", unavailable_integer_mm)
    packed_int4_module._cpu_int_mm_available.cache_clear()
    with pytest.raises(RuntimeError, match="integer kernel is unavailable"):
        resident.score_queries(queries=queries)
    assert torch.equal(
        resident.score_queries(queries=queries, require_integer=False),
        queries.cosine_similarity(gallery),
    )

    def unregistered_integer_mm(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        del left, right
        raise RuntimeError("backend registration wording changed")

    monkeypatch.setattr(torch, "_int_mm", unregistered_integer_mm)
    packed_int4_module._cpu_int_mm_available.cache_clear()
    assert torch.equal(
        resident.score_queries(queries=queries, require_integer=False),
        queries.cosine_similarity(gallery),
    )

    real_integer_mm = torch.ops.aten._int_mm.default

    def probe_then_oom(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        if left.shape == (1, 2) and right.shape == (2, 1):
            return cast(torch.Tensor, real_integer_mm(left, right))
        raise RuntimeError("out of memory")

    monkeypatch.setattr(torch, "_int_mm", probe_then_oom)
    packed_int4_module._cpu_int_mm_available.cache_clear()
    with pytest.raises(RuntimeError, match="out of memory"):
        resident.score_queries(queries=queries)

    class BackendRuntimeError(RuntimeError):
        pass

    def probe_then_subclass_error(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        if left.shape == (1, 2) and right.shape == (2, 1):
            return cast(torch.Tensor, real_integer_mm(left, right))
        raise BackendRuntimeError("allocator failed")

    monkeypatch.setattr(torch, "_int_mm", probe_then_subclass_error)
    packed_int4_module._cpu_int_mm_available.cache_clear()
    with pytest.raises(BackendRuntimeError, match="allocator failed"):
        resident.score_queries(queries=queries)


def test_packed_int4_rejects_noncanonical_shapes_codes_norms_and_wire() -> None:
    packed = pack_int4_unit_embeddings(_unit(4, 128))
    with pytest.raises(ValueError, match="packed int4 byte authority"):
        PackedInt4Embeddings.from_bytes(b"short", count=1, dimensions=128)
    with pytest.raises(ValueError, match="packed int4 byte authority"):
        PackedInt4Embeddings.from_bytes(packed.to_bytes(), count=4, dimensions=127)
    with pytest.raises(ValueError, match="packed int4 embedding authority"):
        PackedInt4Embeddings.from_bytes(
            bytes((0x88,)) + struct.pack("<e", 1.0), count=1, dimensions=2
        )
    with pytest.raises(ValueError, match="packed int4 embedding authority"):
        PackedInt4Embeddings(
            packed_codes=torch.tensor([[0x08]], dtype=torch.uint8),
            inverse_norms=torch.ones(1, dtype=torch.float16),
            dimensions=2,
        )
    with pytest.raises(ValueError, match="packed int4 embedding authority"):
        PackedInt4Embeddings(
            packed_codes=packed.packed_codes,
            inverse_norms=torch.ones(4, dtype=torch.float16),
            dimensions=128,
        )
    with pytest.raises(ValueError, match="quantization authority"):
        pack_int4_unit_embeddings(torch.zeros((2, 128), dtype=torch.float32))


def test_packed_int4_accepts_interoperable_one_ulp_inverse_norm() -> None:
    packed = pack_int4_unit_embeddings(_unit(4, 128))
    alternative = packed.inverse_norms.clone()
    alternative[0] = torch.nextafter(alternative[0], torch.tensor(torch.inf, dtype=torch.float16))

    value = PackedInt4Embeddings(
        packed_codes=packed.packed_codes,
        inverse_norms=alternative,
        dimensions=128,
    )

    assert torch.equal(value.inverse_norms, alternative)


def test_relational_linear_trainer_is_deterministic_and_returns_finite_history() -> None:
    source = _unit(8, 4)
    teacher = _unit(8, 5).roll(1, dims=0)
    basis = torch.eye(4, dtype=torch.float32)[:2]
    config = RelationalLinearTrainingConfig(
        batch_size=4,
        epochs=2,
        learning_rate=1e-3,
        seed=17,
        temperature=0.1,
        weight_decay=1e-4,
    )

    first, first_losses = fit_relational_linear_encoder(
        source, teacher, basis, config=config, device=torch.device("cpu")
    )
    second, second_losses = fit_relational_linear_encoder(
        source, teacher, basis, config=config, device=torch.device("cpu")
    )

    assert first_losses == second_losses
    assert len(first_losses) == config.epochs
    assert all(torch.isfinite(torch.tensor(first_losses)))
    torch.testing.assert_close(first.projection.weight, second.projection.weight)


def test_relational_linear_trainer_detaches_caller_autograd_graphs() -> None:
    source = _unit(8, 4).requires_grad_()
    teacher = _unit(8, 5).roll(1, dims=0).requires_grad_()
    model, losses = fit_relational_linear_encoder(
        source,
        teacher,
        torch.eye(4, dtype=torch.float32)[:2],
        config=RelationalLinearTrainingConfig(batch_size=4, epochs=2),
        device=torch.device("cpu"),
    )

    assert len(losses) == 2
    assert source.grad is None
    assert teacher.grad is None
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_relational_linear_trainer_reduces_neighborhood_loss() -> None:
    generator = torch.Generator().manual_seed(7)
    source = F.normalize(torch.randn(16, 4, generator=generator), dim=1)
    teacher = F.normalize(source @ torch.randn(4, 5, generator=generator), dim=1)
    basis = torch.eye(4, dtype=torch.float32)[:2]
    initial = neighborhood_distribution_kl(
        RelationalLinearEncoder(basis)(source), teacher, temperature=0.2
    )

    model, _losses = fit_relational_linear_encoder(
        source,
        teacher,
        basis,
        config=RelationalLinearTrainingConfig(
            batch_size=16,
            epochs=30,
            learning_rate=0.03,
            temperature=0.2,
            weight_decay=0.0,
        ),
        device=torch.device("cpu"),
    )
    final = neighborhood_distribution_kl(model(source), teacher, temperature=0.2)

    assert float(final.detach()) < float(initial.detach())


def test_relational_linear_compaction_fits_its_train_only_basis() -> None:
    source = _unit(8, 4)
    teacher = _unit(8, 5).roll(1, dims=0)

    model, losses = fit_relational_linear_compaction(
        source,
        teacher,
        output_dimensions=2,
        config=RelationalLinearTrainingConfig(batch_size=4, epochs=2),
        device=torch.device("cpu"),
    )

    assert model.projection.weight.shape == (2, 4)
    assert len(losses) == 2


def test_relational_linear_training_config_rejects_invalid_recipe() -> None:
    with pytest.raises(ValueError, match="training config"):
        RelationalLinearTrainingConfig(batch_size=1)
    with pytest.raises(ValueError, match="training config"):
        RelationalLinearTrainingConfig(batch_size=2)


def test_base_package_import_does_not_require_optional_torch() -> None:
    source = """
import sys
class RejectTorch:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise ModuleNotFoundError('torch is intentionally unavailable')
sys.meta_path.insert(0, RejectTorch())
import sfora
assert 'RelationalLinearEncoder' in sfora.__all__
"""
    result = subprocess.run(
        [sys.executable, "-c", source],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_relational_linear_method_is_available_from_public_api() -> None:
    from sfora import (
        PackedInt4Embeddings as PublicPackedInt4,
    )
    from sfora import (
        PackedInt8Embeddings as PublicPacked,
    )
    from sfora import (
        RelationalLinearEncoder as PublicEncoder,
    )
    from sfora import (
        RelationalLinearTrainingConfig as PublicConfig,
    )
    from sfora import (
        ResidentInt4Gallery as PublicResidentInt4Gallery,
    )
    from sfora import (
        fit_relational_linear_compaction as public_compact_fit,
    )
    from sfora import (
        fit_relational_linear_encoder as public_fit,
    )
    from sfora import (
        pack_int4_unit_embeddings as public_pack_int4,
    )
    from sfora import (
        pack_int8_unit_embeddings as public_pack,
    )

    assert PublicPackedInt4 is PackedInt4Embeddings
    assert PublicPacked is PackedInt8Embeddings
    assert PublicEncoder is RelationalLinearEncoder
    assert PublicConfig is RelationalLinearTrainingConfig
    assert PublicResidentInt4Gallery is ResidentInt4Gallery
    assert public_compact_fit is fit_relational_linear_compaction
    assert public_fit is fit_relational_linear_encoder
    assert public_pack_int4 is pack_int4_unit_embeddings
    assert public_pack is pack_int8_unit_embeddings
