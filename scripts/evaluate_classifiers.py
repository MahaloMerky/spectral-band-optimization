"""Re-score band sets with classifiers on held-out scenes (the wrapper check).

For each fold (stats_fold<i>.npz names its training and test scenes), trains
LDA, a random forest and a small MLP on the training scenes' pixels and reports
macro-F1 and per-class F1 on the held-out scenes, for these feature sets:

  exact k   the fold's exact-search selection (<results-dir>/exact_fold<i>/k<k>/)
  OLI       the seven reflective OLI bands, simulated as in evaluate_sensors.py
  all       all candidate bands, as an upper reference
  extra     any other band sets given with --bands NAME=nm,nm,...

    python scripts/evaluate_classifiers.py --pixels data/processed/pixels.npz \
        --stats-dir data/processed/stats --results-dir results --k 3 4 \
        --out results/classifiers.csv
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from bandopt.labels import CLASSES
from bandopt.sensors import OLI_BAND_EDGES_NM, OLI_REFLECTIVE, band_weights

CLASSIFIERS = {
    "LDA": lambda: make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()),
    "RF": lambda: RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0),
    "MLP": lambda: make_pipeline(StandardScaler(), MLPClassifier(
        hidden_layer_sizes=(64, 64), max_iter=400, early_stopping=True, random_state=0)),
}


def selection(results_dir, fold, k):
    path = results_dir / f"exact_fold{fold}" / f"k{k}" / "top_subsets.csv"
    if not path.exists():
        return None
    with path.open() as file:
        return [int(i) for i in next(csv.DictReader(file))["band_indices"].split()]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pixels", type=Path, required=True)
    parser.add_argument("--stats-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--k", type=int, nargs="+", default=[3, 4])
    parser.add_argument("--classifiers", nargs="+", default=list(CLASSIFIERS))
    parser.add_argument("--bands", nargs="*", default=[], help="Extra sets, e.g. learned=450,560,870")
    parser.add_argument("--no-all", action="store_true", help="Skip the all-bands reference")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    table = np.load(args.pixels, allow_pickle=True)
    labels, scenes = table["labels"], table["scene_ids"].astype(str)
    rows = []
    for stats_path in sorted(args.stats_dir.glob("stats_fold*.npz")):
        fold = int(stats_path.stem.replace("stats_fold", ""))
        stats = np.load(stats_path, allow_pickle=True)
        candidates = stats["wavelengths_nm"]
        columns = np.searchsorted(table["wavelengths_nm"], candidates)
        spectra = table["spectra"][:, columns]  # the fold's candidate bands
        train = np.isin(scenes, stats["train_scenes"].astype(str))
        test = np.isin(scenes, stats["test_scenes"].astype(str))

        feature_sets = {}
        for k in args.k:
            chosen = selection(args.results_dir, fold, k)
            if chosen is None:
                print(f"fold {fold}: no exact k={k} result, skipped")
            else:
                feature_sets[f"exact k={k}"] = spectra[:, chosen]
        oli = band_weights(candidates, [OLI_BAND_EDGES_NM[n] for n in OLI_REFLECTIVE]).numpy()
        feature_sets["OLI (7)"] = spectra @ oli.T
        for spec in args.bands:
            name, values = spec.split("=")
            chosen = [int(np.abs(candidates - float(v)).argmin()) for v in values.split(",")]
            feature_sets[name] = spectra[:, chosen]
        if not args.no_all:
            feature_sets[f"all ({len(candidates)})"] = spectra

        present = np.unique(labels[test])
        for features_name, features in feature_sets.items():
            for classifier_name in args.classifiers:
                model = CLASSIFIERS[classifier_name]().fit(features[train], labels[train])
                predicted = model.predict(features[test])
                per_class = f1_score(labels[test], predicted, labels=present, average=None, zero_division=0)
                row = {
                    "fold": fold, "features": features_name, "n_features": features.shape[1],
                    "classifier": classifier_name,
                    "macro_f1": f"{f1_score(labels[test], predicted, labels=present, average='macro', zero_division=0):.4f}",
                    "test_pixels": int(test.sum()),
                }
                row.update({f"f1_{CLASSES[c]}": f"{v:.4f}" for c, v in zip(present, per_class)})
                rows.append(row)
                print(f"fold {fold} {features_name:12} {classifier_name:4} macro-F1 {row['macro_f1']}", flush=True)

    fields = list(dict.fromkeys(key for row in rows for key in row))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
