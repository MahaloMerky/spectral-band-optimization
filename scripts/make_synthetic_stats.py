"""Write random class statistics for trying the pipeline without real data."""

import argparse
from pathlib import Path

import numpy as np

from bandopt.stats import class_statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--classes", type=int, default=7)
    parser.add_argument("--bands", type=int, default=40)
    parser.add_argument("--pixels-per-class", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    centres = rng.uniform(0.05, 0.6, size=(args.classes, args.bands))
    spectra = np.concatenate(
        [c + rng.normal(0, 0.03, size=(args.pixels_per_class, args.bands)) for c in centres]
    )
    labels = np.repeat(np.arange(args.classes), args.pixels_per_class)
    mu, sigma, _ = class_statistics(spectra, labels)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    wavelengths = np.linspace(400, 2500, args.bands)
    np.savez(args.out, mu=mu, sigma=sigma, wavelengths_nm=wavelengths)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
