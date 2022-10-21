from typing import Optional

import torch
from torch import Tensor
from torch.nn import functional as F


def evidential_uncertainty_loss(
    evidence: Tensor,
    target: Tensor,
    reg_factor: float = 0.0,
    reduction: Optional[str] = "mean",
):
    r"""Computes the evidential uncertainty loss
    Args:
        evidence (Tensor): Evidence of model output.  The evidence is any non-negative
            transformation math:`g(x)` of the raw unnormalized model outputs
            math:`x`.  For example `g=relu(x)`, `g=exp(x)` etc.
        target (Tensor): Ground truth class indicies or class probabilities.
        reg_factor (float): The regularization factor.  If 0, no regularization is
            performed.  If > 0, the regularization term is added to the loss with
            weight loss + reg_factor * reg_term.
        reduction (str): The reduction type.  Must be one of
            ['mean', 'sum', 'none', None]. If None or 'none' no reduction
            is performed. Default is "mean".

    Returns:
        (Tensor) The mean kl regularized dirichlet mse loss over the batch.
    """
    loss = dirichlet_mse_loss(evidence, target, reduction=reduction)
    if reg_factor > 0:
        kl_loss = uniform_dirichlet_kl(evidence, target, reduction=reduction)
        loss += reg_factor * kl_loss

    return loss


def dirichlet_mse_loss(
    evidence: Tensor, target: Tensor, reduction: Optional[str] = "mean"
) -> Tensor:
    r"""This criterion computes the mean dirirchlet mse loss between the input
    and target.

    Reference:
        * Evidential Deep Learning for Quantifying Classification Uncertainty
        * https://arxiv.org/pdf/1806.01768.pdf

    Args:
        evidence (Tensor): Evidence of model output.  The evidence is any non-negative
            transformation math:`g(x)` of the raw unnormalized model outputs
            math:`x`.  For example `g=exp(x)`, `g=relu(x)` etc.
        target (Tensor): Ground truth class indicies or class probabilities.
        reduction (str): The reduction type.  Must be one of
            ['mean', 'sum', 'none', None]. If None or 'none' no reduction
            is performed. Default is "mean".

    Returns:
        (Tensor) The reduced dirichlet mse loss over the batch
    """
    alpha = evidence + 1  # (batch_size, num_clases)
    s_alpha = torch.sum(alpha, dim=1, keepdim=True)  # (batch_size,1)
    p_hat = alpha / s_alpha  # (batch_size, num_clases), \hat{p_ij} from paper
    target = enfore_same_dim(target, evidence)

    error_term = torch.square(target - p_hat)

    var_term = p_hat * (1 - p_hat) / (s_alpha + 1)  # (batch_size, num_classes)
    mse = torch.sum(error_term + var_term, dim=1)
    reducer = get_reducer(reduction)
    return reducer(mse)


def uniform_dirichlet_kl(
    evidence: Tensor, target: Tensor, reduction: Optional[str] = "mean"
) -> Tensor:
    r"""This criterion computes the mean Kullback-Leibler divergence from the
    uniform Dirichlet against the incorrect classes.

    Implementation Note:
        In general the beta term below should be
        `beta_term = torch.sum(lgamma(beta),dim=1, keepdim=True) - lgamma(beta)`
        but the first term (the sum) is 0 since beta is a vector of ones so we
        omit it.
    Reference:
        http://bariskurt.com/kullback-leibler-divergence-between-two-dirichlet-and-beta-distributions/

    Args:
        evidence (Tensor): Evidence of model output.  The evidence is any non-negative
            transformation math:`g(x)` of the raw unnormalized model outputs
            math:`x`.  For example `g=relu(x)`, `g=exp(x)` etc.
        target (Tensor): Ground truth class indicies or class probabilities.
        reduction (str): The reduction type.  Must be one of
            ['mean', 'sum', 'none', None]. If None or 'none' no reduction
            is performed. Default is "mean".

    Returns:
        (Tensor) The mean kl loss from the uniform dirichlet over the batch in each
        entry except the correct class entry.

    """
    lgamma = torch.lgamma
    digamma = torch.digamma
    device = evidence.device

    alpha = evidence + 1  # ( batch_size, num_classes)
    beta = torch.ones((1, *evidence.shape[1:]), device=device)  # (1, num_classes)

    target = enfore_same_dim(target, evidence)
    alpha = target + (1 - target) * alpha  # alpha_hat from the paper

    # sums
    s_alpha = torch.sum(alpha, dim=1, keepdim=True)  # (batch_size, 1)
    s_beta = torch.sum(beta, dim=1)  # (batch_size, d1, d2, ..., dn)

    # compute the terms contributing to final sum
    alpha_term = lgamma(s_alpha) - torch.sum(lgamma(alpha), dim=1)
    beta_term = -lgamma(s_beta)
    digamma_term = (alpha - beta) * (digamma(alpha) - digamma(s_alpha))
    digamma_term = torch.sum(digamma_term, dim=1)

    reducer = get_reducer(reduction)
    return reducer(alpha_term + beta_term + digamma_term)


def maxnorm_uncertainty_loss(
    evidence: Tensor,
    target: Tensor,
    reg_factor: float = 0.0,
    p_norm: int = 4,
    reduction: Optional[str] = "mean",
) -> Tensor:
    r"""Computes the max norm uncertainty loss.

    Args:
        evidence (Tensor): Evidence of model output.  The evidence is any non-negative
            transformation math:`g(x)` of the raw unnormalized model outputs
            math:`x`.  For example `g=relu(x)`, `g=exp(x)` etc.
        target (Tensor): Ground truth class indicies or class probabilities.
        reg_factor (float): The weight of regularization factor.  If 0,
            no regularization is applied.
        p_norm (int): The p-norm to use for the max norm approximation.
        reduction (str): The reduction type.  Must be one of
            ['mean', 'sum', 'none', None]. If None or 'none' no reduction
            is performed. Default is "mean".

    Returns:
        (Tensor) The max norm uncertainty loss over the batch.
    """
    loss = dirichlet_pnorm_loss(evidence, target, p_norm=p_norm, reduction=reduction)
    if reg_factor > 0:
        fisher_loss = dirichlet_fisher_regulizer(evidence, target, reduction=reduction)
        loss += reg_factor * fisher_loss
    return loss


def dirichlet_pnorm_loss(
    evidence: Tensor, target: Tensor, p_norm=4, reduction: Optional[str] = "mean"
) -> Tensor:
    r"""This criterion computes the mean dirichlet max norm loss between the input
    and target.

    The implementation is performed in log space so we use torch's `lgamma` and
    to avoid numerical issues.  We first compute the log of the loss function and
    then exponentiate it to get the loss.  As usual prods are replaced with sums
    and divisons are replaced with subtractions and sums are replaced with logsumexp.
    It may be helpful to understand this implementation by writing down the mathematics
    of log(F) where F is the loss function from the paper.

    Reference:
        * Information Aware Max-Norm Dirichlet Networks for Predictive Uncertainty Estimation # noqa
        * https://arxiv.org/pdf/1910.04819.pdf

    Args:
        evidence (Tensor): Evidence of model output.  The evidence is any non-negative
            transformation math:`g(x)` of the raw unnormalized model outputs
            math:`x`.  For example `g=relu(x)`, `g=exp(x)` etc.
        target (Tensor): Ground truth class indicies or class probabilities.
        p_norm (int): The p-norm to use for the max norm approximation.
        reduction (str): The reduction type.  Must be one of
            ['mean', 'sum', 'none', None]. If None or 'none' no reduction
            is performed. Default is "mean".

    Returns:
        (Tensor) The mean dirichlet max norm loss over the batch
    """
    p = p_norm
    lgamma = torch.lgamma
    target = enfore_same_dim(target, evidence)

    alpha = evidence + 1

    # mask out correct class in alpha
    mask = target > 0
    alpha_hat = alpha.clone()
    alpha_hat[mask] = 0

    # prepare to sum over all indicies except the correct class
    #
    # lgamma(1) = 0, so the below ensures that when we take
    # sum(alpha + p) we get 0 in the correct class entry which
    # which is what we want since we really should be summing over
    # all classes except the correct class.
    alpha_negp = alpha.clone()
    alpha_negp[mask] = -p + 1

    s_alpha = torch.sum(alpha, dim=1, keepdim=True)
    s_alpha_hat = torch.sum(alpha_hat, dim=1, keepdim=True)

    factored_term = torch.squeeze(lgamma(s_alpha) - lgamma(s_alpha + p))
    logsumexp_term_0 = lgamma(s_alpha_hat + p) - lgamma(s_alpha_hat)
    logsumexp_term_1 = lgamma(alpha_negp + p) - lgamma(alpha_hat)

    lse = torch.cat([logsumexp_term_0, logsumexp_term_1], dim=1)
    logsumexp_term = torch.logsumexp(lse, dim=1)

    loss = torch.exp((factored_term + logsumexp_term) / p)
    reducer = get_reducer(reduction)
    return reducer(loss)


def dirichlet_fisher_regulizer(
    evidence: Tensor, target: Tensor, reduction: Optional[str] = "mean"
) -> Tensor:
    r"""This criterion computes the mean dirichlet fisher regulizer between the input
    and target.

    Reference:
        * Information Aware Max-Norm Dirichlet Networks for Predictive Uncertainty Estimation # noqa
        * https://arxiv.org/pdf/1910.04819.pdf

    Args:
        evidence (Tensor): Evidence of model output.  The evidence is any non-negative
            transformation math:`g(x)` of the raw unnormalized model outputs
            math:`x`.  For example `g=relu(x)`, `g=exp(x)` etc.
        target (Tensor): Ground truth class indicies or class probabilities.
        reduction (str): The reduction type.  Must be one of
            ['mean', 'sum', 'none', None]. If None or 'none' no reduction
            is performed. Default is "mean".

    Returns:
        (Tensor) The mean dirichlet fisher regulizer over the batch
    """
    target = enfore_same_dim(target, evidence)
    alpha = evidence + 1
    alpha_hat = alpha.clone()
    mask = target > 0

    # prepare to sum over all indicies except the correct class
    #
    # we need sum(alpha-1) over all classes except the correct class
    # so we do the subtraction and then set the correct class entry to 0
    alpha_minus_one = alpha.clone()
    alpha_minus_one = alpha_minus_one - 1
    alpha_minus_one[mask] = 0

    s_alpha_hat = torch.sum(alpha_hat, dim=1, keepdim=True)
    polygamma_term = torch.polygamma(1, alpha_hat) - torch.polygamma(1, s_alpha_hat)
    prod = torch.square(alpha_minus_one) * polygamma_term
    reducer = get_reducer(reduction)
    return reducer(0.5 * torch.sum(prod, dim=1))


def get_reducer(reduction: Optional[str] = None):
    """Returns a reducer function for the given reduction type.

    Args:
        reduction (str): The reduction type.  Must be one of
            ['mean', 'sum', 'none', None]. If None or 'none' no reduction
            is performed. Default is None.
    Returns:
        (Callable) A reducer function.
    """
    if reduction is None:
        return lambda x: x
    elif reduction == "mean":
        return torch.mean
    elif reduction == "sum":
        return torch.sum
    elif reduction == "none":
        return lambda x: x
    else:
        raise ValueError(
            f"reduction must be one of 'mean', 'sum', 'none' or None, got {reduction}"
        )


def enfore_same_dim(target: Tensor, evidence: Tensor) -> Tensor:
    """Enforces that target is one-hot encoded so that it is
    the same numer of dimensions as the evidence tensor.

    This is required to handle the case where the evidence tensor
    is shape (batch, num_classes, d_1, ... d_k) and the target
    tensor is sparse encoded as shape (batch, d_1, ... d_k).

    Args:
        target (Tensor): The target tensor.

    Returns:
        (Tensor) The one hot encoded target tensor.
    """
    if target.dim() != evidence.dim():
        target = F.one_hot(
            target.reshape(target.size(0), -1), num_classes=evidence.shape[1]
        )
        # (batch, prod(d1,...,dk), classes) -> (batch, classes, d1,...,dk)
        target = torch.transpose(target, 1, 2).reshape(evidence.shape)

    return target
