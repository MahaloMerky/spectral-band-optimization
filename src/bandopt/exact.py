"""Exhaustive scoring of a range of band subsets."""

from dataclasses import dataclass

import torch

from bandopt.combinatorics import binomial_table, unrank_subsets
from bandopt.separability import worst_case_jm


@dataclass
class ChunkResult:
    """The best subsets found in a rank range, best first."""

    ranks: torch.Tensor
    scores: torch.Tensor
    n_failed: int


@torch.inference_mode()
def search_rank_range(mu, sigma, k, start, stop, batch_size=100_000, top=100):
    """Score ranks in [start, stop) and keep the highest-scoring subsets.

    Subsets that could not be scored are counted in n_failed. A nonzero count
    means the result is not a guaranteed optimum.
    """
    device = mu.device
    n_bands = mu.shape[1]
    table = binomial_table(n_bands, k, device=device)
    best_ranks = torch.empty(0, dtype=torch.int64)
    best_scores = torch.empty(0, dtype=mu.dtype)
    n_failed = 0

    for batch_start in range(start, stop, batch_size):
        batch_stop = min(batch_start + batch_size, stop)
        ranks = torch.arange(batch_start, batch_stop, dtype=torch.int64, device=device)
        subsets = unrank_subsets(ranks, n_bands, k, table)
        scores = worst_case_jm(mu, sigma, subsets)

        valid = ~scores.isnan()
        n_failed += int((~valid).sum())
        candidate_ranks = torch.cat([best_ranks, ranks[valid].cpu()])
        candidate_scores = torch.cat([best_scores, scores[valid].cpu()])
        keep = torch.topk(candidate_scores, min(top, candidate_scores.numel())).indices
        best_ranks = candidate_ranks[keep]
        best_scores = candidate_scores[keep]

    return ChunkResult(best_ranks, best_scores, n_failed)
