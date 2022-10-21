import pytest

torch = pytest.importorskip("torch")
from uncertainty_loss._torch import (  # noqa
    dirichlet_fisher_regulizer,
    dirichlet_mse_loss,
    dirichlet_pnorm_loss,
    maxnorm_uncertainty_loss,
    uniform_dirichlet_kl,
)


@pytest.mark.parametrize(
    "reduction,batch,expected",
    [(None, 10, (10,)), ("none", 10, (10,)), ("mean", 10, ()), ("sum", 10, ())],
)
def test_dirichlet_mse_loss_returns_correct_shape_with_reduction(
    reduction, batch, expected
):
    """Verifies the shape of the mse loss tensor is correct."""
    y_true = torch.rand(batch, 3)
    y_pred = torch.rand(batch, 3)
    loss = dirichlet_mse_loss(y_true, y_pred, reduction=reduction)
    assert loss.shape == expected


@pytest.mark.parametrize("one_hot", [True, False])
def test_dirichlet_mse_loss_returns_correct_value(one_hot):
    """Verifies the value of the mse loss tensor is small when there is
    a large amount of evidence for hte correct class.
    """
    if one_hot:
        y_true = torch.tensor([[0, 0, 1]])
    else:
        y_true = torch.tensor([2])
    y_pred = torch.tensor([[0, 0, 1000]])
    loss = dirichlet_mse_loss(y_pred, y_true)
    assert loss.item() < 1e-5


@pytest.mark.parametrize("one_hot", [True, False])
def test_uniform_kl_regularization_is_zero_when_no_evidence_for_wrong_class(one_hot):
    """Verifies the uniform kl reguarlization term is small when there is
    no evidence for the wrong class."""
    if one_hot:
        y_true = torch.tensor([[0, 0, 1]])
    else:
        y_true = torch.tensor([2])
    y_pred = torch.tensor([[0, 0, 1000]])
    loss = uniform_dirichlet_kl(y_pred, y_true)
    assert loss.item() == 0


@pytest.mark.parametrize("one_hot", [True, False])
def test_uniform_kl_regularization_is_not_zero_when_evidence_for_wrong_class(one_hot):
    """Verifies the uniform kl reguarlization term is not zero when there is
    evidence for the wrong class."""
    if one_hot:
        y_true = torch.tensor([[0, 0, 1]])
    else:
        y_true = torch.tensor([2])
    y_pred = torch.tensor([[1, 1, 1000]])
    loss = uniform_dirichlet_kl(y_pred, y_true)
    assert loss.item() > 0


@pytest.mark.parametrize(
    "reduction,batch,expected",
    [(None, 10, (10,)), ("none", 10, (10,)), ("mean", 10, ()), ("sum", 10, ())],
)
def test_dirichlet_pnorm_loss_returns_correct_shape_with_reduction(
    reduction, batch, expected
):
    """Verifies the shape of the mse loss tensor is correct."""
    y_true = torch.rand(batch, 3)
    y_pred = torch.rand(batch, 3)
    loss = dirichlet_pnorm_loss(y_true, y_pred, reduction=reduction)
    assert loss.shape == expected


@pytest.mark.parametrize(
    "reduction,batch,expected",
    [(None, 10, (10,)), ("none", 10, (10,)), ("mean", 10, ()), ("sum", 10, ())],
)
def test_maxnorm_loss_returns_correct_shape_with_reduction(reduction, batch, expected):
    """Verifies the shape of the mse loss tensor is correct."""
    y_true = torch.rand(batch, 3)
    y_pred = torch.rand(batch, 3)
    loss = maxnorm_uncertainty_loss(y_true, y_pred, reg_factor=1.0, reduction=reduction)
    assert loss.shape == expected


@pytest.mark.parametrize("one_hot", [True, False])
def test_fisher_regularization_is_zero_when_no_evidence_for_wrong_class(one_hot):
    """Verifies the fisher reguarlization term is small when there is
    no evidence for the wrong class."""
    if one_hot:
        y_true = torch.tensor([[0, 0, 1]])
    else:
        y_true = torch.tensor([2])
    y_pred = torch.tensor([[0, 0, 1000]])
    loss = dirichlet_fisher_regulizer(y_pred, y_true)
    assert loss.item() == 0


@pytest.mark.parametrize("one_hot", [True, False])
def test_fisher_regularization_is_not_zero_when_evidence_for_wrong_class(one_hot):
    """Verifies the fisher reguarlization term is not zero when there is
    evidence for the wrong class."""
    if one_hot:
        y_true = torch.tensor([[0, 0, 1]])
    else:
        y_true = torch.tensor([2])
    y_pred = torch.tensor([[1, 1, 1000]])
    loss = dirichlet_fisher_regulizer(y_pred, y_true)
    assert loss.item() > 0
