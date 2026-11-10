"""Batched Bhattacharyya and Jeffries-Matusita separability (Eqs. 2-3).

The searches score millions of tiny k x k problems at once. General LAPACK and
cuSOLVER routines carry per-call overhead that dominates at this size, so the
Cholesky factorisation and the triangular solve are unrolled over the matrix
entries. Each matrix entry is held as its own contiguous tensor over the whole
batch, so every step is one element-wise operation with sequential memory
access, which runs well on both CPU and GPU.
"""

import torch


def _cholesky_entries(entries):
    """Factor a batch of small symmetric matrices given as separate entries.

    Args:
        entries: entries[i][j] for j <= i, each a tensor of the same shape
            (...,), holding element (i, j) of every matrix in the batch.

    Returns:
        Factor entries in the same layout, log-determinants (...,), and a mask
        (...,) that is True where the matrix is positive definite. Factors of
        failed matrices are finite but meaningless.
    """
    k = len(entries)
    factors = [[None] * (i + 1) for i in range(k)]
    inverse_diagonal = [None] * k
    ok = torch.ones_like(entries[0][0], dtype=torch.bool)
    log_det = torch.zeros_like(entries[0][0])
    for j in range(k):
        pivot = entries[j][j].clone()
        for m in range(j):
            pivot.sub_(factors[j][m] * factors[j][m])
        positive = pivot > 0
        ok &= positive
        diagonal = torch.sqrt(torch.where(positive, pivot, torch.ones_like(pivot)))
        factors[j][j] = diagonal
        log_det.add_(torch.log(diagonal), alpha=2)
        inverse_diagonal[j] = 1 / diagonal
        for i in range(j + 1, k):
            value = entries[i][j].clone()
            for m in range(j):
                value.sub_(factors[i][m] * factors[j][m])
            factors[i][j] = value.mul_(inverse_diagonal[j])
    return factors, log_det, ok


def _cholesky(matrices):
    """Matrix-shaped wrapper of _cholesky_entries, for shape (..., k, k)."""
    k = matrices.shape[-1]
    entries = [[matrices[..., i, j] for j in range(i + 1)] for i in range(k)]
    factor_entries, log_det, ok = _cholesky_entries(entries)
    factors = torch.zeros_like(matrices)
    for i in range(k):
        for j in range(i + 1):
            factors[..., i, j] = factor_entries[i][j]
    return factors, log_det, ok


def _whitened_norm(factors, vectors):
    """Return |L^-1 v|^2 by forward substitution, with entries as in _cholesky_entries."""
    solved = []
    norm = torch.zeros_like(vectors[0])
    for i, vector in enumerate(vectors):
        value = vector.clone()
        for m in range(i):
            value.sub_(factors[i][m] * solved[m])
        value.div_(factors[i][i])
        solved.append(value)
        norm.add_(value * value)
    return norm


def jm_from_bhattacharyya(distance):
    """Convert Bhattacharyya distance to Jeffries-Matusita distance."""
    return -2 * torch.expm1(-distance)


def pairwise_bhattacharyya(mu, sigma, subsets):
    """Compute the Bhattacharyya distance of every class pair for a batch of subsets.

    Args:
        mu: Class means with shape (classes, bands).
        sigma: Class covariances with shape (classes, bands, bands).
        subsets: Band indices with shape (batch, k).

    Returns:
        Distances with shape (batch, pairs), ordered by torch.triu_indices.
        A subset is NaN for any pair whose covariance is not positive definite.
    """
    n_classes = mu.shape[0]
    k = subsets.shape[1]
    first, second = torch.triu_indices(n_classes, n_classes, offset=1, device=mu.device)
    columns = [subsets[:, i] for i in range(k)]

    # Gather only the lower triangle, one contiguous (classes, batch) tensor per entry.
    class_entries = [
        [sigma[:, columns[i], columns[j]] for j in range(i + 1)] for i in range(k)
    ]
    _, class_log_det, class_ok = _cholesky_entries(class_entries)

    mean_entries = [
        [0.5 * (entry[first] + entry[second]) for entry in row] for row in class_entries
    ]
    factors, mean_log_det, mean_ok = _cholesky_entries(mean_entries)
    differences = [mu[first][:, column] - mu[second][:, column] for column in columns]
    mahalanobis = _whitened_norm(factors, differences)

    distance = mahalanobis / 8 + 0.5 * (
        mean_log_det - 0.5 * (class_log_det[first] + class_log_det[second])
    )
    valid = class_ok[first] & class_ok[second] & mean_ok
    distance = torch.where(valid, distance, torch.full_like(distance, torch.nan))
    return distance.T  # (batch, pairs)


def pairwise_jm(mu, sigma, subsets):
    """Compute the JM distance of every class pair for a batch of subsets."""
    return jm_from_bhattacharyya(pairwise_bhattacharyya(mu, sigma, subsets))


def worst_case_bhattacharyya(mu, sigma, subsets):
    """Compute the smallest pairwise Bhattacharyya distance for each subset.

    JM increases with Bhattacharyya distance, so this ranks subsets the same way
    as F(S) in Eq. 3, without JM's rounding ties close to 2. Returns NaN for a
    subset if any of its pairs could not be evaluated.
    """
    distances = pairwise_bhattacharyya(mu, sigma, subsets)
    scores = distances.nan_to_num(torch.inf).amin(dim=1)
    return torch.where(distances.isnan().any(dim=1), torch.nan, scores)


def worst_case_jm(mu, sigma, subsets):
    """Compute F(S), the smallest pairwise JM distance, for each subset (Eq. 3)."""
    return jm_from_bhattacharyya(worst_case_bhattacharyya(mu, sigma, subsets))
