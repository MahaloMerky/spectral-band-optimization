"""Exhaustively score every k-band subset on one PC, in resumable chunks.

The search space is split into chunks of --chunk-size ranks. Each finished chunk
is saved to its own file, so an interrupted run picks up where it stopped when
started again with the same arguments. Use --chunks to run only part of the
search, for example to split it between two PCs, then copy all chunk files into
one directory and run merge_exact.py.
"""

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from bandopt.combinatorics import chunk_range
from bandopt.exact import (
    describe_device,
    pick_device,
    search_rank_range,
    statistics_fingerprint,
)


def parse_chunk_selection(text, n_chunks):
    """Parse "all", "5", or "0-31" into a range of chunk ids."""
    if text == "all":
        return range(n_chunks)
    first, _, last = text.partition("-")
    first = int(first)
    last = int(last) if last else first
    if not 0 <= first <= last < n_chunks:
        raise SystemExit(f"--chunks {text} is outside 0-{n_chunks - 1}")
    return range(first, last + 1)


def finished(path, expected):
    """Return True if a chunk file exists and matches this search."""
    if not path.exists():
        return False
    with np.load(path) as chunk:
        found = {key: chunk[key].item() for key in expected if key in chunk.files}
    if found != expected:
        raise SystemExit(
            f"{path} comes from a different search ({found} vs {expected}). "
            "Use another --out directory or delete the old chunks."
        )
    return True


def save_atomically(path, **arrays):
    """Write an .npz file so that an interrupted write never leaves a partial chunk."""
    partial = path.with_name(path.stem + ".partial.npz")
    np.savez(partial, **arrays)
    os.replace(partial, path)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:1, or cpu")
    parser.add_argument("--threads", type=int, help="CPU threads (default: PyTorch's choice)")
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--chunk-size", type=int, default=50_000_000, help="Ranks per saved chunk")
    parser.add_argument("--chunks", default="all", help='Chunk ids to run: "all", "7", or "0-31"')
    parser.add_argument("--top", type=int, default=1000, help="Subsets kept per chunk")
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    device = pick_device(args.device)

    stats = np.load(args.stats)
    mu_np, sigma_np = stats["mu"], stats["sigma"]
    mu = torch.as_tensor(mu_np, dtype=torch.float64, device=device)
    sigma = torch.as_tensor(sigma_np, dtype=torch.float64, device=device)
    n_bands = mu.shape[1]

    total = math.comb(n_bands, args.k)
    n_chunks = math.ceil(total / args.chunk_size)
    selected = parse_chunk_selection(args.chunks, n_chunks)
    fingerprint = statistics_fingerprint(mu_np, sigma_np)
    args.out.mkdir(parents=True, exist_ok=True)

    print(
        f"{total:,} subsets of {args.k} from {n_bands} bands in {n_chunks} chunks; "
        f"running chunks {selected.start}-{selected.stop - 1} on {describe_device(device)}"
    )
    done_ranks, done_seconds = 0, 0.0
    remaining = sum(
        stop - start
        for start, stop in (chunk_range(total, n_chunks, c) for c in selected)
    )

    for chunk_id in selected:
        start, stop = chunk_range(total, n_chunks, chunk_id)
        path = args.out / f"chunk_{chunk_id:05d}.npz"
        expected = {
            "start": start, "stop": stop, "n_bands": n_bands, "k": args.k,
            "top": args.top, "stats_fingerprint": fingerprint,
        }
        if finished(path, expected):
            remaining -= stop - start
            continue

        began = time.perf_counter()
        result = search_rank_range(mu, sigma, args.k, start, stop, args.batch_size, args.top)
        elapsed = time.perf_counter() - began
        save_atomically(
            path,
            ranks=result.ranks.numpy(),
            scores=result.scores.numpy(),
            n_failed=result.n_failed,
            elapsed_seconds=elapsed,
            device=describe_device(device),
            **expected,
        )

        done_ranks += stop - start
        done_seconds += elapsed
        remaining -= stop - start
        rate = done_ranks / done_seconds if done_seconds else float("inf")
        eta = remaining / rate if rate else 0.0
        print(
            f"chunk {chunk_id}/{n_chunks - 1}: {elapsed:.1f} s, {rate:,.0f} subsets/s, "
            f"about {eta / 60:.1f} min left",
            flush=True,
        )

    print(f"Done. Merge with: python scripts/merge_exact.py {args.out} --stats {args.stats}")


if __name__ == "__main__":
    main()
