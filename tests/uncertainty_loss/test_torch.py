import pytest

torch = pytest.importorskip("torch")
from uncertainty_loss._torch import (  # noqa
    dirichlet_fisher_regulizer,
    dirichlet_mse_loss,
    dirichlet_pnorm_loss,
    maxnorm_uncertainty_loss,
    uniform_dirichlet_kl,
)


def make_correct_example(one_hot: bool, dims: int = 3):
    """Creates test data for the loss functions

    The test data is a 2d tensor of shape (batch_size, classes) or
    a 3d tensor of shape (batch, classes, 2) or a 4d
    tensor of shape (batch, classes, 2, 2). This allows us to test
    that the loss functions properly handle the case where the input
    tensor is shape (N,C,d1,..,dk) and the target tensor is shape
    (N,C,d1,..,dk) or (N,d1,...,dk).
    """
    # 3 class, 2d data (classes first)
    evidence = torch.tensor(
        [
            [[0, 0], [0, 0]],
            [[0, 1000], [0, 1000]],
            [[1000, 0], [1000, 0]],
        ]
    ).reshape(1, 3, 2, 2)
    if one_hot:
        y_true = torch.tensor(
            [
                [[0, 0], [0, 0]],
                [[0, 1], [0, 1]],
                [[1, 0], [1, 0]],
            ]
        ).reshape(1, 3, 2, 2)
    else:
        y_true = torch.tensor([[[2, 1], [2, 1]]]).reshape(1, 2, 2)

    if dims == 2:
        evidence = evidence[:, :, 0, 0]
        if one_hot:
            y_true = y_true[:, :, 0, 0]
        else:
            y_true = y_true[:, 0, 0]
    elif dims == 3:
        evidence = evidence[:, :, 0]
        if one_hot:
            y_true = y_true[:, :, 0]
        else:
            y_true = y_true[:, 0]

    return evidence, y_true


def make_incorrect_example(one_hot: bool, dims: int = 3):
    """Creates test data for the loss functions

    The test data is a 2d tensor of shape (batch_size, classes) or
    a 3d tensor of shape (batch, classes, 2) or a 4d
    tensor of shape (batch, classes, 2, 2). This allows us to test
    that the loss functions properly handle the case where the input
    tensor is shape (N,C,d1,..,dk) and the target tensor is shape
    (N,C,d1,..,dk) or (N,d1,...,dk).
    """
    # 3 class, 2d data (classes first)
    evidence = torch.tensor(
        [
            [[0, 0], [0, 0]],
            [[0, 1000], [0, 1000]],
            [[1000, 0], [1000, 0]],
        ]
    ).reshape(1, 3, 2, 2)
    if one_hot:
        y_true = torch.tensor(
            [
                [[0, 0], [0, 0]],
                [[1, 0], [1, 0]],
                [[0, 1], [0, 1]],
            ]
        ).reshape(1, 3, 2, 2)
    else:
        y_true = torch.tensor([[[1, 2], [1, 2]]]).reshape(1, 2, 2)

    if dims == 2:
        evidence = evidence[:, :, 0, 0]
        if one_hot:
            y_true = y_true[:, :, 0, 0]
        else:
            y_true = y_true[:, 0, 0]
    elif dims == 3:
        evidence = evidence[:, :, 0]
        if one_hot:
            y_true = y_true[:, :, 0]
        else:
            y_true = y_true[:, 0]

    return evidence, y_true


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
@pytest.mark.parametrize("dims", [2, 3, 4])
def test_dirichlet_mse_loss_returns_correct_value(one_hot, dims):
    """Verifies the value of the mse loss tensor is small when there is
    a large amount of evidence for hte correct class.
    """
    evidence, y_true = make_correct_example(one_hot, dims)
    loss = dirichlet_mse_loss(evidence, y_true)
    assert loss.item() < 1e-5


@pytest.mark.parametrize("one_hot", [True, False])
@pytest.mark.parametrize("dims", [2, 3, 4])
def test_uniform_kl_regularization_is_zero_when_no_evidence_for_wrong_class(
    one_hot, dims
):
    """Verifies the uniform kl reguarlization term is small when there is
    no evidence for the wrong class."""
    y_pred, y_true = make_correct_example(one_hot, dims)
    loss = uniform_dirichlet_kl(y_pred, y_true)
    assert loss.item() == 0


@pytest.mark.parametrize("one_hot", [True, False])
@pytest.mark.parametrize("dims", [2, 3, 4])
def test_uniform_kl_regularization_is_not_zero_when_evidence_for_wrong_class(
    one_hot, dims
):
    """Verifies the uniform kl reguarlization term is not zero when there is
    evidence for the wrong class."""
    y_pred, y_true = make_incorrect_example(one_hot, dims)
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
@pytest.mark.parametrize("dims", [2, 3, 4])
def test_fisher_regularization_is_zero_when_no_evidence_for_wrong_class(one_hot, dims):
    """Verifies the fisher reguarlization term is small when there is
    no evidence for the wrong class."""
    y_pred, y_true = make_correct_example(one_hot, dims)
    loss = dirichlet_fisher_regulizer(y_pred, y_true)
    assert loss.item() == 0


@pytest.mark.parametrize("one_hot", [True, False])
@pytest.mark.parametrize("dims", [2, 3, 4])
def test_fisher_regularization_is_not_zero_when_evidence_for_wrong_class(one_hot, dims):
    """Verifies the fisher reguarlization term is not zero when there is
    evidence for the wrong class."""
    y_pred, y_true = make_incorrect_example(one_hot, dims)
    loss = dirichlet_fisher_regulizer(y_pred, y_true)
    assert loss.item() > 0
