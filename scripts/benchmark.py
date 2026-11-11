"""Measure exact-search throughput on this PC and project full search times.

Run it on each candidate PC and compare the subsets/s column, for example:
    python scripts/benchmark.py --device cuda
    python scripts/benchmark.py --device cpu

Uses random statistics with the planned problem size, so no data is needed.
Add --log results/benchmark.csv to keep each measurement, for example to compare
thread counts for RQ4:
    python scripts/benchmark.py --device cpu --threads 4 --log results/benchmark.csv
"""

import argparse
import math
import time
from pathlib import Path

import torch

from bandopt.exact import describe_device, pick_device, search_rank_range
from bandopt.timing import append_rows, format_duration, timestamp

BENCHMARK_LOG_FIELDS = (
    "recorded_at", "k", "n_bands", "classes", "batch_size", "subsets", "seconds",
    "subsets_per_s", "projected_seconds", "device",
)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--bands", type=int, default=255)
    parser.add_argument("--classes", type=int, default=7)
    parser.add_argument("--k", type=int, nargs="+", default=[3, 4, 5, 6])
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[25_000, 100_000, 400_000]
    )
    parser.add_argument("--seconds", type=float, default=5.0, help="Time per measurement")
    parser.add_argument("--log", type=Path, help="Append each measurement to this CSV file")
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    device = pick_device(args.device)
    generator = torch.Generator().manual_seed(0)
    mu = torch.rand(args.classes, args.bands, generator=generator, dtype=torch.float64)
    factors = torch.randn(
        args.classes, args.bands, args.bands, generator=generator, dtype=torch.float64
    )
    sigma = factors @ factors.mT / args.bands + 0.05 * torch.eye(args.bands, dtype=torch.float64)
    mu, sigma = mu.to(device), sigma.to(device)

    print(f"Device: {describe_device(device)}; {args.classes} classes, {args.bands} bands")
    print(f"{'k':>2} {'batch':>9} {'subsets/s':>14} {'full search':>14}")
    for k in args.k:
        total = math.comb(args.bands, k)
        for batch_size in args.batch_sizes:
            # Warm up once, then score whole batches until the time budget is spent.
            search_rank_range(mu, sigma, k, 0, min(batch_size, total), batch_size)
            synchronize(device)
            scored, began = 0, time.perf_counter()
            while time.perf_counter() - began < args.seconds:
                start = scored % total
                stop = min(start + batch_size, total)
                search_rank_range(mu, sigma, k, start, stop, batch_size)
                scored += stop - start
            synchronize(device)
            seconds = time.perf_counter() - began
            rate = scored / seconds
            print(f"{k:>2} {batch_size:>9,} {rate:>14,.0f} {format_duration(total / rate):>14}")
            if args.log:
                append_rows(args.log, [{
                    "recorded_at": timestamp(),
                    "k": k,
                    "n_bands": args.bands,
                    "classes": args.classes,
                    "batch_size": batch_size,
                    "subsets": scored,
                    "seconds": f"{seconds:.6f}",
                    "subsets_per_s": f"{rate:.1f}",
                    "projected_seconds": f"{total / rate:.1f}",
                    "device": describe_device(device),
                }], BENCHMARK_LOG_FIELDS)


if __name__ == "__main__":
    main()
