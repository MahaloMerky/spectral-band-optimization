"""Run every search and experiment after the statistics exist, in order.

Steps (each skipped if its output already exists, unless --force):
  1. Exact search and merge: every fold and all scenes for k in --k-every,
     and --k-split (default 5) on the --split-fold only. Searches resume from
     their saved chunks, so an interrupted run can simply be started again.
  2. OLI comparison for all scenes and every fold.
  3. Heuristics (SFS, SFFS, mRMR, group lasso, SA, GA) on all scenes.
  4. Classifier check on held-out scenes.
  5. Differentiable sensor design.
  6. Paper figures and tables.

    python scripts/run_all.py --data D:/bandopt-data/screened --device cuda

--data is the folder holding processed/pixels.npz and processed/stats/;
results go to <data>/results/.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(arguments, log):
    command = [sys.executable, "-u", *map(str, arguments)]
    print(f"\n$ {' '.join(command[2:])}", flush=True)
    with log.open("a") as file:
        file.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(command[2:])}\n")
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode:
        sys.exit(f"Step failed with exit code {result.returncode}: {' '.join(command[2:])}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--device", default="auto", help="For the exact search: auto, cuda or cpu")
    parser.add_argument("--k-every", type=int, nargs="+", default=[3, 4], help="k searched on every fold")
    parser.add_argument("--k-split", type=int, nargs="*", default=[5], help="k searched on one fold only")
    parser.add_argument("--split-fold", default="fold0")
    parser.add_argument("--heuristic-k", type=int, nargs="+", default=[3, 4, 5, 6, 8, 10])
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--skip", nargs="*", default=[],
                        choices=["exact", "sensors", "heuristics", "classifiers", "differentiable", "figures"])
    parser.add_argument("--force", action="store_true", help="Redo steps whose outputs exist")
    args = parser.parse_args()

    stats_dir = args.data / "processed" / "stats"
    pixels = args.data / "processed" / "pixels.npz"
    results = args.data / "results"
    results.mkdir(parents=True, exist_ok=True)
    log = results / "run_all.log"
    stats_names = ["all"] + sorted(p.stem.replace("stats_", "") for p in stats_dir.glob("stats_fold*.npz"))
    if not (stats_dir / "stats_all.npz").exists():
        sys.exit(f"No statistics in {stats_dir}; run compute_stats.py first")

    def todo(path):
        return args.force or not Path(path).exists()

    if "exact" not in args.skip:
        jobs = [(name, k) for k in args.k_every for name in stats_names]
        jobs += [(args.split_fold, k) for k in args.k_split]
        for name, k in jobs:
            out = results / f"exact_{name}" / f"k{k}"
            stats = stats_dir / f"stats_{name}.npz"
            if todo(out / "top_subsets.csv"):
                run(["scripts/exact_search.py", "--stats", stats, "--k", k, "--out", out, "--device", args.device], log)
                run(["scripts/merge_exact.py", out, "--stats", stats], log)

    if "sensors" not in args.skip:
        for name in stats_names:
            out = results / f"sensors_{name}.csv"
            if todo(out):
                run(["scripts/evaluate_sensors.py", "--stats", stats_dir / f"stats_{name}.npz",
                     "--exact-dir", results / f"exact_{name}", "--out", out], log)

    if "heuristics" not in args.skip and todo(results / "heuristics_all.csv"):
        run(["scripts/run_heuristics.py", "--stats", stats_dir / "stats_all.npz", "--pixels", pixels,
             "--k", *args.heuristic_k, "--seeds", args.seeds, "--exact-dir", results / "exact_all",
             "--out", results / "heuristics_all.csv"], log)

    if "classifiers" not in args.skip and todo(results / "classifiers.csv"):
        run(["scripts/evaluate_classifiers.py", "--pixels", pixels, "--stats-dir", stats_dir,
             "--results-dir", results, "--k", *args.k_every, "--out", results / "classifiers.csv"], log)

    if "differentiable" not in args.skip and todo(results / "differentiable.csv"):
        device = "cuda" if args.device == "cuda" else "cpu"
        run(["scripts/train_differentiable.py", "--pixels", pixels, "--stats-dir", stats_dir,
             "--k", 3, 4, 5, "--seeds", 3, "--device", device, "--out", results / "differentiable.csv"], log)

    if "figures" not in args.skip:
        run(["scripts/make_figures.py", "--pixels", pixels, "--stats-dir", stats_dir,
             "--results-dir", results], log)
    print(f"\nAll steps done. Results in {results}; log in {log}")


if __name__ == "__main__":
    main()
