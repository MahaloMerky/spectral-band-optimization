"""Map integers to band subsets so a search space splits into array tasks."""

import math

import torch

INT64_MAX = 2**63 - 1


def binomial_table(n, k, device=None):
    """Return C(c, j) for c in 0..n-1 and j in 0..k, with shape (k + 1, n)."""
    if math.comb(n, k) > INT64_MAX:
        raise ValueError(f"C({n}, {k}) does not fit in a 64-bit integer")
    values = [[math.comb(c, j) for c in range(n)] for j in range(k + 1)]
    return torch.tensor(values, dtype=torch.int64, device=device)


def unrank_subsets(ranks, n, k, table=None):
    """Convert ranks to subsets with the combinatorial number system.

    Args:
        ranks: Integer tensor with values in [0, C(n, k)) and shape (batch,).
        n: Number of candidate bands.
        k: Subset size.
        table: Optional output of binomial_table, to avoid rebuilding it.

    Returns:
        Band indices with shape (batch, k), ascending within each row.
    """
    if table is None:
        table = binomial_table(n, k, device=ranks.device)
    remaining = ranks.clone()
    subsets = torch.empty((ranks.shape[0], k), dtype=torch.int64, device=ranks.device)
    for j in range(k, 0, -1):
        column = table[j]
        chosen = torch.searchsorted(column, remaining, right=True) - 1
        subsets[:, j - 1] = chosen
        remaining -= column[chosen]
    return subsets


def chunk_range(total, n_tasks, task_id):
    """Return the half-open rank range for one task, with near-equal chunk sizes."""
    if not 0 <= task_id < n_tasks:
        raise ValueError(f"task_id must be in [0, {n_tasks}), got {task_id}")
    return task_id * total // n_tasks, (task_id + 1) * total // n_tasks
