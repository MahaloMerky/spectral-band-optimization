"""Learn band centres and widths by gradient descent (RQ3).

For each fold (stats_fold<i>.npz names its training and test scenes) and each
k, trains a LearnableSensor + classifier on the training scenes' full EMIT
spectra from several random starts, then reports for every start the learned
centres and FWHMs, macro-F1 on the held-out scenes, and the worst-case JM of the
learned bands on the fold's class statistics (comparable with the exact search).

    python scripts/train_differentiable.py --pixels data/processed/pixels.npz \
        --stats-dir data/processed/stats --k 3 4 5 --seeds 5 \
        --out results/differentiable.csv
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from bandopt.differentiable import predict, train
from bandopt.labels import CLASSES
from bandopt.sensors import FWHM_PER_SIGMA, gaussian_response
from bandopt.separability import pairwise_jm


def learned_jm(stats, centres, fwhm):
    """Worst-case and mean JM of Gaussian bands on a statistics file's candidate grid."""
    grid = torch.as_tensor(stats["wavelengths_nm"], dtype=torch.float64)
    responses = gaussian_response(grid, torch.as_tensor(centres, dtype=torch.float64),
                                  torch.as_tensor(fwhm, dtype=torch.float64) / FWHM_PER_SIGMA)
    weights = responses / responses.sum(dim=1, keepdim=True)
    mu = torch.as_tensor(stats["mu"], dtype=torch.float64) @ weights.T
    sigma = weights @ torch.as_tensor(stats["sigma"], dtype=torch.float64) @ weights.T
    jm = pairwise_jm(mu, sigma, torch.arange(len(centres))[None])[0]
    return float(jm.min()), float(jm.mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pixels", type=Path, required=True)
    parser.add_argument("--stats-dir", type=Path, required=True)
    parser.add_argument("--k", type=int, nargs="+", default=[3, 4, 5])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--alpha", type=float, default=1.0, help="Absorption penalty weight")
    parser.add_argument("--beta", type=float, default=1e-3, help="Minimum-width penalty weight")
    parser.add_argument("--min-fwhm", type=float, default=10.0, help="nm")
    parser.add_argument("--noise", type=float, default=0.005, help="Reflectance noise at 30 nm FWHM")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--folds", nargs="*", type=int, help="Only these folds")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    table = np.load(args.pixels, allow_pickle=True)
    spectra, labels = table["spectra"], table["labels"]
    scenes, wavelengths = table["scene_ids"].astype(str), table["wavelengths_nm"]
    rows = []
    for stats_path in sorted(args.stats_dir.glob("stats_fold*.npz")):
        fold = int(stats_path.stem.replace("stats_fold", ""))
        if args.folds and fold not in args.folds:
            continue
        stats = np.load(stats_path, allow_pickle=True)
        train_mask = np.isin(scenes, stats["train_scenes"].astype(str))
        test_mask = np.isin(scenes, stats["test_scenes"].astype(str))
        present = np.unique(labels[test_mask])
        for k in args.k:
            for seed in range(args.seeds):
                began = time.perf_counter()
                model, history = train(
                    spectra[train_mask], labels[train_mask], wavelengths, k, len(CLASSES), seed=seed,
                    epochs=args.epochs, alpha=args.alpha, beta=args.beta, min_fwhm_nm=args.min_fwhm,
                    noise=args.noise, device=args.device,
                )
                seconds = time.perf_counter() - began
                predicted = predict(model, spectra[test_mask], device=args.device)
                per_class = f1_score(labels[test_mask], predicted, labels=present, average=None, zero_division=0)
                macro = f1_score(labels[test_mask], predicted, labels=present, average="macro", zero_division=0)
                order = torch.argsort(model.sensor.centres).tolist()
                centres = model.sensor.centres.detach().numpy()[order]
                fwhm = model.sensor.fwhm.detach().numpy()[order]
                worst, mean = learned_jm(stats, centres, fwhm)
                row = {
                    "fold": fold, "k": k, "seed": seed,
                    "centres_nm": " ".join(f"{c:.1f}" for c in centres),
                    "fwhm_nm": " ".join(f"{w:.1f}" for w in fwhm),
                    "macro_f1": f"{macro:.4f}", "min_jm": f"{worst:.4f}", "mean_jm": f"{mean:.4f}",
                    "final_loss": f"{history['loss'][-1]:.4f}", "seconds": f"{seconds:.1f}",
                }
                row.update({f"f1_{CLASSES[c]}": f"{v:.4f}" for c, v in zip(present, per_class)})
                rows.append(row)
                print(f"fold {fold} k={k} seed {seed}: centres {row['centres_nm']} nm, FWHM {row['fwhm_nm']} nm, "
                      f"held-out macro-F1 {row['macro_f1']}, worst-case JM {row['min_jm']} ({seconds:.0f} s)",
                      flush=True)

    # Random starts can stall in a local optimum. Mark, per fold and k, the start
    # with the lowest training loss; held-out scores play no part in the choice.
    for row in rows:
        group = [r for r in rows if r["fold"] == row["fold"] and r["k"] == row["k"]]
        row["chosen"] = int(row is min(group, key=lambda r: float(r["final_loss"])))
    fields = list(dict.fromkeys(key for row in rows for key in row))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
