"""Exhaustive scoring of a range of band subsets."""

import hashlib
from dataclasses import dataclass

import numpy as np
import torch

from bandopt.combinatorics import binomial_table, unrank_subsets
from bandopt.separability import worst_case_bhattacharyya


@dataclass
class ChunkResult:
    """The best subsets found in a rank range, best first, on the CPU.

    Scores are worst-case Bhattacharyya distances.
    """

    ranks: torch.Tensor
    scores: torch.Tensor
    n_failed: int


def pick_device(name="auto"):
    """Return a torch device: CUDA when available for "auto", else the named one."""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but this PyTorch build cannot see a GPU. "
            "Install a CUDA build of PyTorch (see README)."
        )
    return torch.device(name)


def describe_device(device):
    """Return a short human-readable name for a device."""
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return f"CPU, {torch.get_num_threads()} threads"


def statistics_fingerprint(mu, sigma):
    """Return a short hash that identifies a statistics file's contents."""
    digest = hashlib.sha256()
    for array in (mu, sigma):
        digest.update(np.ascontiguousarray(array, dtype=np.float64).tobytes())
    return digest.hexdigest()[:16]


@torch.inference_mode()
def search_rank_range(mu, sigma, k, start, stop, batch_size=100_000, top=1000, on_batch=None):
    """Score ranks in [start, stop) and keep the highest-scoring subsets.

    Subsets that could not be scored are counted in n_failed. A nonzero count
    means the result is not a guaranteed optimum. If given, on_batch is called
    with the number of subsets in each batch once it has been queued, for
    progress reporting.
    """
    device = mu.device
    n_bands = mu.shape[1]
    table = binomial_table(n_bands, k, device=device)
    best_ranks = torch.empty(0, dtype=torch.int64, device=device)
    best_scores = torch.empty(0, dtype=mu.dtype, device=device)
    n_failed = torch.zeros((), dtype=torch.int64, device=device)

    for batch_start in range(start, stop, batch_size):
        batch_stop = min(batch_start + batch_size, stop)
        ranks = torch.arange(batch_start, batch_stop, dtype=torch.int64, device=device)
        subsets = unrank_subsets(ranks, n_bands, k, table)
        scores = worst_case_bhattacharyya(mu, sigma, subsets)

        failed = scores.isnan()
        n_failed += failed.sum()
        # Failed subsets drop to -inf so they never enter the top list.
        scores = scores.masked_fill(failed, -torch.inf)
        candidate_ranks = torch.cat([best_ranks, ranks])
        candidate_scores = torch.cat([best_scores, scores])
        keep = torch.topk(candidate_scores, min(top, candidate_scores.numel())).indices
        best_ranks = candidate_ranks[keep]
        best_scores = candidate_scores[keep]
        if on_batch is not None:
            on_batch(batch_stop - batch_start)

    finite = best_scores > -torch.inf
    return ChunkResult(
        best_ranks[finite].cpu(), best_scores[finite].cpu(), int(n_failed.item())
    )
