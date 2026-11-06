"""Score one chunk of the exhaustive search space. Run once per array task."""

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from bandopt.combinatorics import chunk_range
from bandopt.exact import search_rank_range


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument("--task-id", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    parser.add_argument("--n-tasks", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--top", type=int, default=1000)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    stats = np.load(args.stats)
    mu = torch.as_tensor(stats["mu"], dtype=torch.float64, device=device)
    sigma = torch.as_tensor(stats["sigma"], dtype=torch.float64, device=device)

    total = math.comb(mu.shape[1], args.k)
    start, stop = chunk_range(total, args.n_tasks, args.task_id)

    began = time.perf_counter()
    result = search_rank_range(mu, sigma, args.k, start, stop, args.batch_size, args.top)
    elapsed = time.perf_counter() - began

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"chunk_{args.task_id:05d}.npz"
    np.savez(
        path,
        ranks=result.ranks.numpy(),
        scores=result.scores.numpy(),
        n_failed=result.n_failed,
        start=start,
        stop=stop,
        n_bands=mu.shape[1],
        k=args.k,
        top=args.top,
        elapsed_seconds=elapsed,
    )
    rate = (stop - start) / elapsed if elapsed else float("inf")
    print(f"{path}: ranks [{start}, {stop}) on {device}, {elapsed:.2f} s, {rate:,.0f} subsets/s")


if __name__ == "__main__":
    main()
