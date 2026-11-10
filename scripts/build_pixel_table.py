"""Build the labelled TOA reflectance pixel table from EMIT scenes.

For every scene in the config file:
  1. Label cells of the EMIT GLT grid. Surface classes need at least --purity of
     the cell to be one WorldCover class and no L2A cloud, dilated cloud, cirrus,
     or spacecraft flag. Cloud classes need a hand-drawn polygon and the L2A cloud flag.
  2. Map labelled cells back to sensor pixels. A sensor pixel that several cells
     give different labels is dropped, and duplicates are kept once.
  3. Sample up to --max-per-class pixels per class, convert their radiance to TOA
     reflectance, and drop any with non-finite values.

Writes an .npz that scripts/compute_stats.py reads, and a CSV of pixel counts per
scene and class. See configs/scenes.example.toml for the config format.
"""

import argparse
import csv
import glob
import sys
import tomllib
from pathlib import Path

import numpy as np

from bandopt import emit
from bandopt.labels import (
    CLASSES,
    DEFAULT_WORLDCOVER_CLASSES,
    UNLABELLED,
    class_index,
    label_grid,
    read_cloud_polygons,
    worldcover_fractions,
)
from bandopt.toa import (
    W_PER_M2_TO_UW_PER_CM2,
    band_solar_irradiance,
    load_solar_spectrum,
    radiance_to_toa_reflectance,
)

WAVELENGTH_TOLERANCE_NM = 0.5


def resolve(base, path):
    path = Path(path)
    return path if path.is_absolute() else base / path


def parse_mapping(table):
    mapping = {}
    for code, name in table.items():
        if name != "ignore":
            class_index(name)
            mapping[int(code)] = name
    return mapping


def load_config(path):
    with open(path, "rb") as file:
        config = tomllib.load(file)
    base = Path(path).parent
    tiles = sorted(
        {match for pattern in config["worldcover_tiles"]
         for match in glob.glob(str(resolve(base, pattern)))}
    )
    if not tiles:
        sys.exit(f"No WorldCover tiles match {config['worldcover_tiles']}")
    defaults = dict(DEFAULT_WORLDCOVER_CLASSES)
    defaults.update(parse_mapping(config.get("worldcover_classes", {})))

    scenes = []
    for scene in config["scene"]:
        mapping = dict(defaults)
        for code, name in scene.get("worldcover_classes", {}).items():
            if name == "ignore":
                mapping.pop(int(code), None)
            else:
                mapping.update(parse_mapping({code: name}))
        scenes.append({
            "id": scene["id"],
            "radiance": resolve(base, scene["radiance"]),
            "observation": resolve(base, scene["observation"]),
            "mask": resolve(base, scene["mask"]),
            "clouds": resolve(base, scene["clouds"]) if "clouds" in scene else None,
            "mapping": mapping,
        })
    ids = [scene["id"] for scene in scenes]
    if len(set(ids)) != len(ids):
        sys.exit("Scene ids must be unique")
    return resolve(base, config["solar_spectrum"]), tiles, scenes


def sensor_pixels(grid, labels):
    """Map labelled grid cells to unique sensor pixels.

    Returns arrays of downtrack rows, crosstrack columns, labels, and the grid
    (row, col) of one cell per pixel, for locating it.
    """
    cells = np.flatnonzero(labels != UNLABELLED)
    rows = grid.glt_rows.ravel()[cells]
    cols = grid.glt_cols.ravel()[cells]
    cell_labels = labels.ravel()[cells]
    pixel = rows * (cols.max(initial=0) + 1) + cols

    order = np.lexsort((cell_labels, pixel))
    pixel, cell_labels, cells, rows, cols = (
        a[order] for a in (pixel, cell_labels, cells, rows, cols)
    )
    first = np.r_[True, pixel[1:] != pixel[:-1]]
    last = np.r_[first[1:], True]
    # Labels are sorted within each pixel, so it is consistent if its first and
    # last label agree.
    keep = np.flatnonzero(first)[cell_labels[first] == cell_labels[last]]
    return rows[keep], cols[keep], cell_labels[keep], np.unravel_index(cells[keep], grid.shape)


def process_scene(scene, solar, max_per_class, purity, rng, tiles):
    grid = emit.read_grid(scene["radiance"])
    shape = radiance_shape(scene["radiance"])
    cloud, any_flag = emit.read_cloud_flags(scene["mask"])
    zenith, distance = emit.read_geometry(scene["observation"])
    for name, array in (("mask", cloud), ("observation", zenith)):
        if array.shape != shape:
            sys.exit(f"{scene['id']}: {name} file shape {array.shape} does not match "
                     f"radiance {shape}. Are the files from the same granule?")
    fractions = worldcover_fractions(tiles, grid, scene["mapping"])
    polygons = read_cloud_polygons(scene["clouds"], grid) if scene["clouds"] else None
    labels = label_grid(
        grid, fractions, grid.to_grid(cloud), grid.to_grid(any_flag), polygons, purity
    )
    rows, cols, pixel_labels, (cell_rows, cell_cols) = sensor_pixels(grid, labels)

    chosen = []
    for index in range(len(CLASSES)):
        members = np.flatnonzero(pixel_labels == index)
        if members.size > max_per_class:
            members = rng.choice(members, max_per_class, replace=False)
        chosen.append(np.sort(members))
    chosen = np.concatenate(chosen)
    rows, cols, pixel_labels = rows[chosen], cols[chosen], pixel_labels[chosen]
    lon, lat = (a[cell_rows[chosen], cell_cols[chosen]] for a in grid.cell_centres())

    wavelengths, fwhm = emit.read_wavelengths(scene["radiance"])
    radiance = emit.read_radiance(scene["radiance"])[rows, cols]

    solar_wavelengths, solar_irradiance = solar
    irradiance = band_solar_irradiance(
        solar_wavelengths, solar_irradiance * W_PER_M2_TO_UW_PER_CM2, wavelengths, fwhm
    )
    reflectance = radiance_to_toa_reflectance(
        radiance, irradiance, zenith[rows, cols], distance[rows, cols]
    )
    finite = np.isfinite(reflectance).all(axis=1)
    return {
        "spectra": reflectance[finite].astype(np.float32),
        "labels": pixel_labels[finite],
        "lon": lon[finite],
        "lat": lat[finite],
        "wavelengths": wavelengths,
        "fwhm": fwhm,
        "n_dropped": int((~finite).sum()),
    }


def radiance_shape(path):
    import netCDF4

    with netCDF4.Dataset(path) as dataset:
        return dataset.variables["radiance"].shape[:2]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Output .npz file")
    parser.add_argument("--max-per-class", type=int, default=2000, help="Per scene")
    parser.add_argument("--purity", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scenes", nargs="+", help="Only process these scene ids")
    args = parser.parse_args()

    solar_path, tiles, scenes = load_config(args.config)
    if args.scenes:
        unknown = set(args.scenes) - {scene["id"] for scene in scenes}
        if unknown:
            sys.exit(f"Unknown scene ids: {sorted(unknown)}")
        scenes = [scene for scene in scenes if scene["id"] in args.scenes]
    solar = load_solar_spectrum(solar_path)
    rng = np.random.default_rng(args.seed)

    parts, counts = [], []
    reference_wavelengths = fwhm = None
    for scene in scenes:
        part = process_scene(scene, solar, args.max_per_class, args.purity, rng, tiles)
        if reference_wavelengths is None:
            reference_wavelengths, fwhm = part["wavelengths"], part["fwhm"]
        elif part["wavelengths"].shape != reference_wavelengths.shape or not np.allclose(
            part["wavelengths"], reference_wavelengths, atol=WAVELENGTH_TOLERANCE_NM
        ):
            sys.exit(f"{scene['id']}: band wavelengths differ from the first scene")
        part["scene_id"] = scene["id"]
        parts.append(part)
        row = [int((part["labels"] == i).sum()) for i in range(len(CLASSES))]
        counts.append([scene["id"], *row])
        print(
            f"{scene['id']}: " + ", ".join(f"{n} {c}" for n, c in zip(row, CLASSES) if n)
            + (f" ({part['n_dropped']} dropped as non-finite)" if part["n_dropped"] else ""),
            flush=True,
        )

    labels = np.concatenate([part["labels"] for part in parts])
    if labels.size == 0:
        sys.exit("No labelled pixels found")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        spectra=np.concatenate([part["spectra"] for part in parts]),
        labels=labels,
        scene_ids=np.concatenate(
            [np.full(part["labels"].size, part["scene_id"]) for part in parts]
        ).astype(str),
        lon=np.concatenate([part["lon"] for part in parts]),
        lat=np.concatenate([part["lat"] for part in parts]),
        wavelengths_nm=reference_wavelengths,
        fwhm_nm=fwhm,
        class_names=np.array(CLASSES),
    )

    summary = args.out.with_suffix(".counts.csv")
    with summary.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["scene_id", *CLASSES])
        writer.writerows(counts)
        writer.writerow(["total", *np.sum([row[1:] for row in counts], axis=0).tolist()])
    print(f"Wrote {args.out} ({labels.size} pixels) and {summary}")

    missing = [c for i, c in enumerate(CLASSES) if not (labels == i).any()]
    if missing:
        print(f"WARNING: no pixels for {missing}. compute_stats.py needs every class.")


if __name__ == "__main__":
    main()
