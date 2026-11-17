"""Score Landsat OLI against the exact EMIT optimum (RQ1).

OLI bands are simulated from a statistics file's candidate bands as Gaussians
(sensors.OLI_BAND_EDGES_NM), so their class statistics are W mu and W Sigma W^T.
For each k up to the number of OLI bands it reports the best OLI k-band subset
(exhaustive over OLI's bands), and, where an exact search result exists, the
EMIT optimum for the same k and the gap 1 - F(OLI) / F(EMIT).

    python scripts/evaluate_sensors.py --stats data/processed/stats/stats_all.npz \
        --exact-dir results/exact_all --out results/sensors_all.csv
    # with --cirrus for the 8-band set (use a --cirrus-run statistics file)
"""

import argparse
import csv
import itertools
from pathlib import Path

import numpy as np
import torch

from bandopt.sensors import OLI_BAND_EDGES_NM, OLI_REFLECTIVE, band_weights
from bandopt.separability import pairwise_jm


def exact_optimum(exact_dir, k):
    path = exact_dir / f"k{k}" / "top_subsets.csv" if exact_dir else None
    if not path or not path.exists():
        return None
    with path.open() as file:
        rows = list(csv.DictReader(file))
    best = max(rows, key=lambda r: float(r["min_jm"]))
    return float(best["min_jm"]), best["wavelengths_nm"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--exact-dir", type=Path)
    parser.add_argument("--cirrus", action="store_true", help="Include OLI B9 (needs a --cirrus-run stats file)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    stats = np.load(args.stats)
    names = list(OLI_BAND_EDGES_NM) if args.cirrus else list(OLI_REFLECTIVE)
    weights = band_weights(stats["wavelengths_nm"], [OLI_BAND_EDGES_NM[n] for n in names])
    mu = torch.as_tensor(stats["mu"], dtype=torch.float64) @ weights.T
    sigma = weights @ torch.as_tensor(stats["sigma"], dtype=torch.float64) @ weights.T

    rows = []
    for k in range(1, len(names) + 1):
        subsets = torch.tensor(list(itertools.combinations(range(len(names)), k)))
        jm = pairwise_jm(mu, sigma, subsets)
        worst, mean = jm.min(dim=1).values, jm.mean(dim=1)
        best = int(torch.argmax(worst))
        exact = exact_optimum(args.exact_dir, k)
        row = {
            "k": k, "oli_bands": " + ".join(names[i] for i in subsets[best].tolist()),
            "oli_min_jm": f"{float(worst[best]):.6f}", "oli_mean_jm": f"{float(mean[best]):.6f}",
            "emit_min_jm": "" if exact is None else f"{exact[0]:.6f}",
            "emit_wavelengths_nm": "" if exact is None else exact[1],
            "gap": "" if exact is None else f"{1 - float(worst[best]) / exact[0]:.6f}",
        }
        rows.append(row)
        extra = f"  EMIT optimum {exact[0]:.4f}, gap {row['gap']}" if exact else ""
        print(f"k={k}: best OLI subset {row['oli_bands']}: worst-case JM {row['oli_min_jm']}{extra}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
