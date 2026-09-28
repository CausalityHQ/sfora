import torch
from probe_inshop_teacher_transfer import information, product_sham

from sfora.teacher_anchored_distillation import cross_dimensional_relational_distillation_loss


def test_product_sham_preserves_pairs_but_changes_relational_gradient():
    teacher = torch.nn.functional.normalize(
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.2, 1.0, 0.0],
                [0.3, 0.9, 0.1],
                [0.0, 0.1, 1.0],
                [0.1, 0.2, 0.9],
            ]
        ),
        dim=1,
    )
    sham = product_sham(teacher)
    assert torch.equal(sham[:2], teacher[-2:])
    assert torch.allclose(
        (teacher[::2] * teacher[1::2]).sum(1).roll(1), (sham[::2] * sham[1::2]).sum(1)
    )
    student = torch.nn.Linear(3, 4)
    codes = torch.nn.functional.normalize(student(teacher), dim=1)
    loss = cross_dimensional_relational_distillation_loss(codes, teacher, temperatures=(0.20,))
    alternate = cross_dimensional_relational_distillation_loss(codes, sham, temperatures=(0.20,))
    true = torch.autograd.grad(loss, student.weight, retain_graph=True)[0]
    other = torch.autograd.grad(alternate, student.weight)[0]
    assert torch.isfinite(true).all() and true.norm() > 0
    assert (true - other).norm() > 0
    assert information(teacher, sham)["teacher_sham_kl"] > 1e-4
