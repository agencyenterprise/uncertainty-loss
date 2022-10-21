from typing import Any, Optional

try:
    from torch import Tensor

    HAS_TORCH = True
    import _torch
except ImportError:
    HAS_TORCH = False


def dispatch(
    func_name: str,
    type_,
) -> Any:
    """Dispatches the loss function to the correct implementation.

    Args:
        y_true: Tensor of true labels.
        y_pred: Tensor of predicted labels.
        reduction: Reduction method for the loss. Defaults to None.

    Returns:
        Tensor of the loss value.
    """
    if HAS_TORCH:
        if type == Tensor:
            return getattr(_torch, func_name)
    else:
        raise ValueError("Only pytorch implementations currently supported.")


def dirichlet_mse_loss(
    evidence: Any,
    y_true: Any,
    reduction: Optional[str] = None,
) -> Any:
    """Computes the mean squared error between two dirichlet distributions.

    Args:
        y_true: Tensor of true labels.
        y_pred: Tensor of predicted labels.
        reduction: Reduction method for the loss. Defaults to None.

    Returns:
        Tensor of the loss value.
    """
    dispatch("dirichlet_mse_loss", type(evidence))(
        evidence, y_true, reduction=reduction
    )
