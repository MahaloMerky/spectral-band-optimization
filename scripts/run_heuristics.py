"""Run the heuristic band selectors and compare them with the exact optimum.

For each k, runs SFS, SFFS and mRMR once and simulated annealing (SA) and the
genetic algorithm (GA) once per seed, on one statistics file. With --pixels it
also runs group-lasso logistic regression (GL) on the pixels of the statistics
file's training scenes; GL optimises classification loss, and its bands are then
scored like the others. Where an exact search result exists
(<exact-dir>/k<k>/top_subsets.csv) it reports the optimality gap
1 - F(S_heur) / F(S*), with F the worst-case JM.

    python scripts/run_heuristics.py --stats data/processed/stats/stats_all.npz \
        --pixels data/processed/pixels.npz \
        --k 3 4 5 6 8 10 --seeds 30 --exact-dir results/exact_all \
        --out results/heuristics_all.csv
"""

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch

from bandopt.group_lasso import select_bands
from bandopt.heuristics import DETERMINISTIC, STOCHASTIC, Scorer
from bandopt.labels import CLASSES
from bandopt.separability import jm_from_bhattacharyya


def to_jm(distance):
    return float(jm_from_bhattacharyya(torch.tensor(distance, dtype=torch.float64)))


def exact_optimum(exact_dir, k):
    path = exact_dir / f"k{k}" / "top_subsets.csv" if exact_dir else None
    if not path or not path.exists():
        return None
    with path.open() as file:
        return max(float(row["min_jm"]) for row in csv.DictReader(file))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--k", type=int, nargs="+", default=[3, 4, 5, 6, 8, 10])
    parser.add_argument("--seeds", type=int, default=30, help="Runs of each stochastic method")
    parser.add_argument("--methods", nargs="+",
                        help="Default: all; GL only when --pixels is given")
    parser.add_argument("--pixels", type=Path, help="Pixel table, needed for group lasso (GL)")
    parser.add_argument("--sa-iterations", type=int, default=20_000)
    parser.add_argument("--ga-generations", type=int, default=150)
    parser.add_argument("--exact-dir", type=Path, help="Folder with k<k>/top_subsets.csv")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int)
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    stats = np.load(args.stats, allow_pickle=True)
    wavelengths = stats["wavelengths_nm"]
    scorer = Scorer(stats["mu"], stats["sigma"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    methods = args.methods or list(DETERMINISTIC) + list(STOCHASTIC) + (["GL"] if args.pixels else [])
    if "GL" in methods:
        if not args.pixels:
            parser.error("GL needs --pixels")
        table = np.load(args.pixels, allow_pickle=True)
        train = np.isin(table["scene_ids"].astype(str), stats["train_scenes"].astype(str))
        columns = np.searchsorted(table["wavelengths_nm"], wavelengths)
        gl_features, gl_labels = table["spectra"][train][:, columns], table["labels"][train]

    rows = []
    for k in args.k:
        optimum = exact_optimum(args.exact_dir, k)
        for method in methods:
            results = []  # (seed, subset, score, seconds, evaluations)
            if method == "GL":
                began = time.perf_counter()
                subset, penalty = select_bands(gl_features, gl_labels, len(CLASSES), k)
                score = float(scorer(np.array([subset]))[0])
                results.append((None, subset, score, time.perf_counter() - began, 1))
            elif method in DETERMINISTIC:
                scorer.evaluations = 0
                began = time.perf_counter()
                subset, score = DETERMINISTIC[method](scorer, k)
                results.append((None, subset, score, time.perf_counter() - began, scorer.evaluations))
            elif method == "SA":
                # All seeds run as parallel chains in one call; time and evaluations
                # are shared out evenly.
                scorer.evaluations = 0
                began = time.perf_counter()
                chains = STOCHASTIC["SA"](scorer, k, np.random.default_rng(0),
                                          iterations=args.sa_iterations, chains=args.seeds)
                chains = chains if args.seeds > 1 else [chains]
                seconds = (time.perf_counter() - began) / args.seeds
                evaluations = scorer.evaluations // args.seeds
                results += [(seed, s, v, seconds, evaluations) for seed, (s, v) in enumerate(chains)]
            else:
                for seed in range(args.seeds):
                    scorer.evaluations = 0
                    began = time.perf_counter()
                    subset, score = STOCHASTIC[method](scorer, k, np.random.default_rng(seed),
                                                       generations=args.ga_generations)
                    results.append((seed, subset, score, time.perf_counter() - began, scorer.evaluations))
            for seed, subset, score, seconds, evaluations in results:
                jm = to_jm(score) if math.isfinite(score) else float("nan")
                rows.append({
                    "k": k, "method": method, "seed": "" if seed is None else seed,
                    "min_bhattacharyya": f"{score:.8f}", "min_jm": f"{jm:.8f}",
                    "exact_min_jm": "" if optimum is None else f"{optimum:.8f}",
                    "gap": "" if optimum is None else f"{1 - jm / optimum:.6f}",
                    "seconds": f"{seconds:.3f}", "evaluations": evaluations,
                    "band_indices": " ".join(map(str, subset)),
                    "wavelengths_nm": " ".join(f"{wavelengths[b]:.1f}" for b in subset),
                })
            done = [r for r in rows if r["k"] == k and r["method"] == method]
            best = max(float(r["min_jm"]) for r in done)
            gap = "" if optimum is None else f", gap of best {1 - best / optimum:.4f}"
            print(f"k={k} {method:5}: best JM {best:.4f} over {len(done)} run(s){gap}", flush=True)

    with args.out.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
