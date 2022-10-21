try:
    import torch  # noqa
except ImportError:
    raise ImportError(
        "PyTorch is not installed.  Please install pytorch before using "
        "uncertainy_loss. You can install pytorch with pip: pip install torch "
    ) from None


from ._torch import (
    clamped_exp,
    dirichlet_fisher_regulizer,
    dirichlet_mse_loss,
    dirichlet_pnorm_loss,
    evidential_loss,
    maxnorm_loss,
    uniform_dirichlet_kl,
)

__all__ = [
    "evidential_loss",
    "maxnorm_loss",
    "dirichlet_mse_loss",
    "uniform_dirichlet_kl",
    "dirichlet_fisher_regulizer",
    "dirichlet_pnorm_loss",
    "clamped_exp",
]
