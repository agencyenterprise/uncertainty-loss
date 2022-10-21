from ._torch import (
    dirichlet_fisher_regulizer,
    dirichlet_mse_loss,
    dirichlet_pnorm_loss,
    evidential_uncertainty_loss,
    maxnorm_uncertainty_loss,
    uniform_dirichlet_kl,
)

__all__ = [
    "evidential_uncertainty_loss",
    "maxnorm_uncertainty_loss",
    "dirichlet_mse_loss",
    "uniform_dirichlet_kl",
    "dirichlet_fisher_regulizer",
    "dirichlet_pnorm_loss",
]
