from __future__ import annotations

import pytest
import torch

from sfora.foldable_linear import FoldableLinear


def test_repeated_identity_is_exactly_foldable_and_parameter_matched() -> None:
    model = FoldableLinear(input_dim=128, hidden_dim=384, output_dim=128)
    model.initialize_repeated_identity()
    values = torch.randn(7, 128, generator=torch.Generator().manual_seed(17))

    weight, bias = model.fold()
    observed = model(values)
    folded = torch.nn.functional.linear(values, weight, bias)

    assert sum(parameter.numel() for parameter in model.parameters()) == 98_432
    torch.testing.assert_close(observed, values, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(folded, observed, rtol=2e-6, atol=2e-7)


def test_factorization_has_gradients_and_fold_tracks_updates() -> None:
    model = FoldableLinear(input_dim=2, hidden_dim=4, output_dim=2)
    model.initialize_repeated_identity()
    values = torch.tensor([[1.0, -2.0], [0.5, 0.25]])
    loss = model(values).square().sum()
    loss.backward()

    assert all(parameter.grad is not None for parameter in model.parameters())
    with torch.no_grad():
        model.output.bias.add_(torch.tensor([0.2, -0.1]))
    weight, bias = model.fold()
    torch.testing.assert_close(
        model(values), torch.nn.functional.linear(values, weight, bias), rtol=2e-6, atol=2e-7
    )


@pytest.mark.parametrize(
    ("input_dim", "hidden_dim", "output_dim"),
    ((0, 4, 2), (2, 0, 2), (2, 4, 0), (2, 3, 2), (2, 4, 3)),
)
def test_repeated_identity_rejects_invalid_shape(
    input_dim: int, hidden_dim: int, output_dim: int
) -> None:
    if min(input_dim, hidden_dim, output_dim) <= 0:
        with pytest.raises(ValueError, match="foldable linear shape"):
            FoldableLinear(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        return
    model = FoldableLinear(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    with pytest.raises(ValueError, match="repeated identity"):
        model.initialize_repeated_identity()
