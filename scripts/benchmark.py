"""Measure exact-search throughput on this PC and project full search times.

Run it on each candidate PC and compare the subsets/s column, for example:
    python scripts/benchmark.py --device cuda
    python scripts/benchmark.py --device cpu

Uses random statistics with the planned problem size, so no data is needed.
"""

import argparse
import math
import time

import torch

from bandopt.exact import describe_device, pick_device, search_rank_range


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
            rate = scored / (time.perf_counter() - began)
            print(f"{k:>2} {batch_size:>9,} {rate:>14,.0f} {format_duration(total / rate):>14}")


def format_duration(seconds):
    for unit, size in (("d", 86400), ("h", 3600), ("min", 60)):
        if seconds >= size:
            return f"{seconds / size:.1f} {unit}"
    return f"{seconds:.1f} s"


if __name__ == "__main__":
    main()
