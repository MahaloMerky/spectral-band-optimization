"""Write class statistics for each scene-based fold and for all data.

The input .npz holds TOA reflectance for labelled pixels:
    spectra         (pixels, bands), finite values only
    labels          (pixels,), integer classes
    scene_ids       (pixels,), fixed-width strings
    wavelengths_nm  (bands,)
"""

import argparse
from pathlib import Path

import numpy as np

from bandopt.sensors import (
    CIRRUS_RUN_WINDOWS_NM,
    WATER_VAPOUR_WINDOWS_NM,
    candidate_band_mask,
)
from bandopt.stats import class_statistics, scene_folds


def write_statistics(path, spectra, labels, wavelengths, train_mask, scene_ids):
    mu, sigma, classes = class_statistics(spectra[train_mask], labels[train_mask])
    np.savez(
        path,
        mu=mu,
        sigma=sigma,
        classes=classes,
        wavelengths_nm=wavelengths,
        train_scenes=np.unique(scene_ids[train_mask]),
        test_scenes=np.unique(scene_ids[~train_mask]),
    )
    print(f"Wrote {path}: {mu.shape[0]} classes, {mu.shape[1]} bands")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pixels", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument(
        "--folds", type=int, default=5,
        help="Scene-based folds; 0 writes only stats_all.npz (for quick tests, e.g. one scene)",
    )
    parser.add_argument(
        "--cirrus-run", action="store_true", help="Keep the 1.35-1.42 um window as candidates"
    )
    args = parser.parse_args()

    table = np.load(args.pixels)
    labels = table["labels"]
    scene_ids = table["scene_ids"]
    windows = CIRRUS_RUN_WINDOWS_NM if args.cirrus_run else WATER_VAPOUR_WINDOWS_NM
    mask = candidate_band_mask(table["wavelengths_nm"], windows).numpy()
    spectra = table["spectra"][:, mask]
    wavelengths = table["wavelengths_nm"][mask]

    args.out.mkdir(parents=True, exist_ok=True)
    folds = scene_folds(labels, scene_ids, args.folds) if args.folds else []
    for fold, (train_mask, _) in enumerate(folds):
        path = args.out / f"stats_fold{fold}.npz"
        write_statistics(path, spectra, labels, wavelengths, train_mask, scene_ids)
    # Used only to report the final bands, never to score held-out performance.
    everything = np.ones(labels.size, dtype=bool)
    write_statistics(args.out / "stats_all.npz", spectra, labels, wavelengths, everything, scene_ids)


if __name__ == "__main__":
    main()
