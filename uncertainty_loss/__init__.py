try:
    import torch  # noqa
except ImportError:
    raise ImportError(
        "PyTorch is not installed.  Please install pytorch before using "
        "uncertainy_loss. You can install pytorch with pip: pip install torch "
    ) from None


from ._torch import (
    clamped_exp,
    cross_entropy_uncertainty,
    data_uncertainty,
    dirichlet_fisher_regulizer,
    dirichlet_mse_loss,
    dirichlet_pnorm_loss,
    entropy,
    evidence_to_proba,
    evidential_loss,
    maxnorm_loss,
    model_uncertainty,
    uncertainty,
    uniform_dirichlet_kl,
)

__all__ = [
    "clamped_exp",
    "cross_entropy_uncertainty",
    "data_uncertainty",
    "dirichlet_fisher_regulizer",
    "dirichlet_mse_loss",
    "dirichlet_pnorm_loss",
    "entropy",
    "evidence_to_proba",
    "evidential_loss",
    "maxnorm_loss",
    "model_uncertainty",
    "uncertainty",
    "uniform_dirichlet_kl",
]
