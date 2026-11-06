"""Batched Bhattacharyya and Jeffries-Matusita separability (Eqs. 2-3)."""

import torch


def _cholesky(matrices):
    """Return Cholesky factors, log-determinants, and a success mask."""
    factors, info = torch.linalg.cholesky_ex(matrices)
    log_determinants = 2 * torch.log(torch.diagonal(factors, dim1=-2, dim2=-1)).sum(-1)
    return factors, log_determinants, info == 0


def pairwise_jm(mu, sigma, subsets):
    """Compute the JM distance of every class pair for a batch of band subsets.

    Args:
        mu: Class means with shape (classes, bands).
        sigma: Class covariances with shape (classes, bands, bands).
        subsets: Band indices with shape (batch, k).

    Returns:
        JM distances with shape (batch, pairs), ordered by torch.triu_indices.
        A subset is NaN for any pair whose covariance is not positive definite.
    """
    n_classes = mu.shape[0]
    mu_s = mu[:, subsets]
    sigma_s = sigma[:, subsets[:, :, None], subsets[:, None, :]]
    _, class_log_det, class_ok = _cholesky(sigma_s)

    first, second = torch.triu_indices(n_classes, n_classes, offset=1, device=mu.device)
    distances = []
    for a, b in zip(first.tolist(), second.tolist()):
        factors, mean_log_det, mean_ok = _cholesky(0.5 * (sigma_s[a] + sigma_s[b]))
        difference = (mu_s[a] - mu_s[b]).unsqueeze(-1)
        whitened = torch.linalg.solve_triangular(factors, difference, upper=False)
        mahalanobis = (whitened**2).sum(dim=(-2, -1))
        bhattacharyya = mahalanobis / 8 + 0.5 * (
            mean_log_det - 0.5 * (class_log_det[a] + class_log_det[b])
        )
        jm = 2 * (1 - torch.exp(-bhattacharyya))
        valid = class_ok[a] & class_ok[b] & mean_ok
        distances.append(torch.where(valid, jm, torch.full_like(jm, torch.nan)))
    return torch.stack(distances, dim=1)


def worst_case_jm(mu, sigma, subsets):
    """Compute F(S), the smallest pairwise JM distance, for each subset (Eq. 3).

    Returns NaN for a subset if any of its pairs could not be evaluated.
    """
    distances = pairwise_jm(mu, sigma, subsets)
    scores = distances.nan_to_num(torch.inf).amin(dim=1)
    return torch.where(distances.isnan().any(dim=1), torch.nan, scores)
