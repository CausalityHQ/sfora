import torch

from sfora.image_end_to_end import (
    ECTRState,
    ImageEndToEndConfig,
    _ectr_composite_loss,
    _ectr_delete_mask,
)


class _ToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer4 = torch.nn.Conv2d(3, 4, 3, padding=1)
        self.fc = torch.nn.Linear(4, 8)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.fc(self.layer4(images).mean((2, 3)))


def test_ectr_mask_and_hinges_are_finite_and_differentiable() -> None:
    model = _ToyModel()
    images = torch.randn(6, 3, 8, 8)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    feature_map = model.layer4(images)
    embeddings = torch.nn.functional.normalize(model.fc(feature_map.mean((2, 3))), dim=1)
    mask = _ectr_delete_mask(feature_map, beta=0.85, torch_module=torch)
    assert mask.shape == (6, 1, 8, 8)
    state = ECTRState(torch.nn.functional.normalize(torch.randn(6, 8), dim=1))
    loss = _ectr_composite_loss(
        model,
        images,
        labels,
        embeddings,
        feature_map,
        torch.arange(6),
        state,
        step=21,
        steps_per_epoch=1,
        config=ImageEndToEndConfig(ectr_weight=0.5),
        torch_module=torch,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_ectr_random_control_uses_float_interpolation_mask() -> None:
    model = _ToyModel()
    images = torch.randn(6, 3, 8, 8)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    feature_map = model.layer4(images)
    embeddings = torch.nn.functional.normalize(model.fc(feature_map.mean((2, 3))), dim=1)
    state = ECTRState(torch.nn.functional.normalize(torch.randn(6, 8), dim=1))
    loss = _ectr_composite_loss(
        model,
        images,
        labels,
        embeddings,
        feature_map,
        torch.arange(6),
        state,
        step=21,
        steps_per_epoch=1,
        config=ImageEndToEndConfig(ectr_weight=0.5, ectr_variant="random"),
        torch_module=torch,
    )
    assert torch.isfinite(loss)
