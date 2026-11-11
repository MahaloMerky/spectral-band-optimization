"""Merge exact-search chunks, verify coverage, and break near-ties by mean JM."""

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch

from bandopt.combinatorics import unrank_subsets
from bandopt.exact import statistics_fingerprint
from bandopt.separability import jm_from_bhattacharyya, pairwise_jm
from bandopt.timing import format_duration


def to_jm(distance):
    return jm_from_bhattacharyya(torch.as_tensor(distance, dtype=torch.float64)).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Directory of chunk_*.npz files")
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--top", type=int, default=100, help="Rows to write")
    parser.add_argument(
        "--tolerance", type=float, default=0.001,
        help="Subsets within this worst-case JM of the best count as tied",
    )
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

    stats = np.load(args.stats)
    fingerprint = statistics_fingerprint(stats["mu"], stats["sigma"])
    if any(str(chunk["stats_fingerprint"]) != fingerprint for chunk in chunks):
        sys.exit(f"Some chunks were computed from statistics other than {args.stats}")
    wavelengths = stats["wavelengths_nm"]
    if wavelengths.size != n_bands:
        sys.exit(f"{args.stats} has {wavelengths.size} bands, but the search used {n_bands}")

    ranks = np.concatenate([chunk["ranks"] for chunk in chunks])
    distances = np.concatenate([chunk["scores"] for chunk in chunks])
    subsets = unrank_subsets(torch.as_tensor(ranks), n_bands, k)

    mu = torch.as_tensor(stats["mu"], dtype=torch.float64)
    sigma = torch.as_tensor(stats["sigma"], dtype=torch.float64)
    pairwise = pairwise_jm(mu, sigma, subsets).numpy()
    min_jm = pairwise.min(axis=1)
    mean_jm = pairwise.mean(axis=1)
    if not np.allclose(min_jm, to_jm(distances)):
        sys.exit(f"Scores in the chunks do not match {args.stats}")

    best_distance = distances.max()
    threshold = to_jm(best_distance) - args.tolerance
    tied = min_jm >= threshold
    # Tied subsets come first, by mean JM. The rest follow by worst-case distance.
    order = np.lexsort((-np.where(tied, mean_jm, distances), ~tied))

    # A full chunk whose last kept subset is still tied may have dropped other ties.
    tie_list_complete = not any(
        chunk["ranks"].size == int(chunk["top"])
        and int(chunk["stop"]) - int(chunk["start"]) > int(chunk["top"])
        and to_jm(chunk["scores"].min()) >= threshold
        for chunk in chunks
    )

    path = args.directory / "top_subsets.csv"
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "tied", "min_bhattacharyya", "min_jm", "mean_jm", "band_indices", "wavelengths_nm",
        ])
        for i in order[: args.top]:
            subset = subsets[i].tolist()
            writer.writerow([
                int(tied[i]),
                f"{distances[i]:.8f}",
                f"{min_jm[i]:.8f}",
                f"{mean_jm[i]:.8f}",
                " ".join(map(str, subset)),
                " ".join(f"{wavelengths[band]:.1f}" for band in subset),
            ])

    def describe(i):
        return " ".join(f"{wavelengths[band]:.1f}" for band in subsets[i].tolist())

    print(f"All {total:,} subsets of {k} from {n_bands} bands were scored.")
    compute_seconds = sum(float(chunk["elapsed_seconds"]) for chunk in chunks)
    devices = sorted({str(chunk["device"]) for chunk in chunks})
    print(
        f"Compute time: {format_duration(compute_seconds)} over {len(chunks)} chunks "
        f"({total / max(compute_seconds, 1e-9):,.0f} subsets/s) on {'; '.join(devices)}"
    )
    print(f"Exact optimum: worst-case JM {to_jm(best_distance):.6f} at {describe(distances.argmax())} nm")
    print(f"Near-ties within {args.tolerance} JM: {int(tied.sum())}")
    print(f"Selected by mean JM: {describe(order[0])} nm (mean JM {mean_jm[order[0]]:.6f})")
    if not tie_list_complete:
        print(
            "WARNING: the near-tie list may be incomplete, so the mean-JM choice is not final. "
            "Rerun exact_search.py with a larger --top."
        )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
