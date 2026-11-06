"""Merge exact-search chunks and verify that they cover the whole search space."""

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch

from bandopt.combinatorics import unrank_subsets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Directory of chunk_*.npz files")
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--top", type=int, default=100)
    args = parser.parse_args()

    chunks = [np.load(path) for path in sorted(args.directory.glob("chunk_*.npz"))]
    if not chunks:
        sys.exit(f"No chunks found in {args.directory}")
    chunks.sort(key=lambda chunk: int(chunk["start"]))

    n_bands = int(chunks[0]["n_bands"])
    k = int(chunks[0]["k"])
    total = math.comb(n_bands, k)

    expected_start = 0
    for chunk in chunks:
        if int(chunk["n_bands"]) != n_bands or int(chunk["k"]) != k:
            sys.exit("Chunks come from different searches")
        if int(chunk["start"]) != expected_start:
            sys.exit(f"Missing or overlapping ranks at {expected_start}")
        expected_start = int(chunk["stop"])
    if expected_start != total:
        sys.exit(f"Chunks stop at rank {expected_start}, but the search space has {total}")

    n_failed = sum(int(chunk["n_failed"]) for chunk in chunks)
    if n_failed:
        sys.exit(f"{n_failed} subsets could not be scored, so the optimum is not guaranteed")

    ranks = np.concatenate([chunk["ranks"] for chunk in chunks])
    scores = np.concatenate([chunk["scores"] for chunk in chunks])
    order = np.argsort(-scores, kind="stable")[: args.top]
    ranks, scores = ranks[order], scores[order]

    wavelengths = np.load(args.stats)["wavelengths_nm"]
    if wavelengths.size != n_bands:
        sys.exit(f"{args.stats} has {wavelengths.size} bands, but the search used {n_bands}")
    subsets = unrank_subsets(torch.as_tensor(ranks), n_bands, k).numpy()

    path = args.directory / "top_subsets.csv"
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["score", "band_indices", "wavelengths_nm"])
        for score, subset in zip(scores, subsets):
            writer.writerow([
                f"{score:.8f}",
                " ".join(map(str, subset)),
                " ".join(f"{wavelengths[i]:.1f}" for i in subset),
            ])

    best = " ".join(f"{wavelengths[i]:.1f}" for i in subsets[0])
    print(f"All {total:,} subsets of {k} from {n_bands} bands were scored.")
    print(f"Best worst-case JM: {scores[0]:.6f} at {best} nm")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
