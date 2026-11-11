# Optimizing Satellite Spectral Bands

GGS 426 (Fall 2026) final project: exact, heuristic, and differentiable band
selection for land-cover and cloud discrimination, using NASA EMIT hyperspectral
imagery on a single workstation with an RTX 5080 GPU. See the [proposal](docs/GGS426_Final_Project_Proposal.pdf).

## Research questions

1. **RQ1:** Which k EMIT bands maximise class separability, and how far below that
   optimum are Landsat 8/9 OLI and Landsat Next?
2. **RQ2:** How close do heuristics come to the exact optimum, and how do they scale?
3. **RQ3:** Where does gradient-based sensor design place band centres and widths,
   and is the placement physically interpretable?
4. **RQ4:** How does throughput compare between the RTX 5080 GPU and the same PC's i9
   CPU, and how does CPU throughput scale with threads?

## Changes from the proposal

| Topic | Proposal | Now | Reason |
| --- | --- | --- | --- |
| Input data | L2A surface reflectance | L1B radiance converted to TOA reflectance, for all classes | Surface reflectance retrieval assumes a view of the surface, which does not hold over clouds. One quantity keeps all classes comparable |
| Search score | Worst-case JM | Worst-case Bhattacharyya distance, reported as JM | Same ranking, without JM's rounding ties close to 2 |
| Near-ties | Not specified | Subsets within 0.001 JM of the best are ranked by mean JM | Gives a defined choice among practically equal subsets |
| Leakage | Spatial CV for evaluation | Class statistics and band selection also use training scenes only | Selecting bands with held-out scenes would inflate scores |
| Computing | SLURM cluster job arrays | One workstation: Intel Core i9 with an RTX 5080 GPU | The cluster has no Python. Chunked, resumable runs replace job arrays |
| OLI baseline | 7 bands | 7 bands in the main run, 8 with cirrus in a sensitivity run for k ≤ 4 | The main run removes the window holding OLI's cirrus band |

"Exact optimum" means the exact optimum of the filter score on the training
statistics. Classifier results may rank subsets differently.

Search folds: every fold for k ≤ 4, one fixed scene split for k = 5, and a final run
on all data (`stats_all.npz`) only to report the bands.

Search space with n ≈ 255 candidate bands, to be recomputed from the real band count:

| k | Subsets |
| --- | --- |
| 3 | 2.7×10⁶ |
| 4 | 1.7×10⁸ |
| 5 | 8.6×10⁹ |
| 6 | 3.6×10¹¹ |

## Setup

```bash
conda env create -f environment.yml
conda activate bandopt
pytest
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

The last line should print `True 12.8` on the RTX 5080 PC. If it prints `False`, the
CPU-only PyTorch build was installed; reinstall it with
`pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128`.
RTX 50-series GPUs need PyTorch 2.7 or later built for CUDA 12.8.

After the environment works, record exact versions so runs are reproducible, and
commit the result:

```bash
conda env export --no-builds > environment.lock.yml
```

## Measuring speed

The benchmark uses random statistics of the planned size, so it needs no data. It
prints subsets per second and the projected time of a full search for each k:

```bash
python scripts/benchmark.py --device cuda   # RTX 5080
python scripts/benchmark.py --device cpu    # i9, for the RQ4 comparison
```

## Building the pixel table

`scripts/build_pixel_table.py` turns EMIT granules into the labelled TOA reflectance
table that `compute_stats.py` reads. Copy `configs/scenes.example.toml` to
`configs/scenes.toml` and list each granule's three files (L1B radiance, L1B
observation geometry, L2A mask), then:

```bash
python scripts/build_pixel_table.py --config configs/scenes.toml --out data/processed/pixels.npz
python scripts/compute_stats.py --pixels data/processed/pixels.npz --out data/processed/stats
```

Try one scene first with `--scenes <id>`. Besides the table, the script writes
`pixels.counts.csv` with pixel counts per scene and class.

How pixels are labelled:

- **Surface classes** come from WorldCover. A pixel needs at least 90% of its area in
  one class (`--purity`) and no L2A cloud, dilated cloud, cirrus, or spacecraft flag.
  Shrubland and cropland are left out by default. WorldCover has one bare class, so
  each scene's config says whether it is `bare_soil` or `sand`.
- **Cloud classes** come from polygons you draw by hand, for example in QGIS, and save
  as GeoJSON in longitude/latitude with a `label` property of `liquid_cloud` or
  `ice_cloud`. Only pixels inside a polygon *and* flagged as cloud by the L2A mask are
  used.
- Labels are made on EMIT's lat/lon grid, then mapped back to sensor pixels, so spectra
  are never resampled. A sensor pixel that gets two different labels is dropped.
- At most `--max-per-class` pixels (default 2000) are sampled per class and scene.

## Exact search

The search reads a statistics file holding `mu` (classes x bands), `sigma`
(classes x bands x bands), and `wavelengths_nm`. For real data,
`scripts/compute_stats.py` writes one file per scene-based fold from a table of
labelled TOA pixels (add `--cirrus-run` for the sensitivity run). That table does not
exist yet, so use synthetic statistics to try the pipeline:

```bash
python scripts/make_synthetic_stats.py --out data/processed/synthetic_stats.npz
python scripts/exact_search.py --stats data/processed/synthetic_stats.npz --k 3 --out results/exact/k3
python scripts/merge_exact.py results/exact/k3 --stats data/processed/synthetic_stats.npz
```

The search uses the GPU when PyTorch can see one, or the CPU otherwise (force one with
`--device cuda` or `--device cpu`, and set CPU threads with `--threads`). It saves
each chunk of `--chunk-size` subsets as it finishes. If a run is interrupted, start
the same command again and it continues from the first missing chunk.

While it runs, the last line of the terminal is a progress bar that updates every
half second, also partway through a chunk:

```
k=5  [############------------------]  40.5%  3,500,000,000/8,637,487,551  1,200,000/s  elapsed 10.0 min  left 1.2 h  chunk 71/172
```

A resumed run starts at its real percentage, and the rate and time left use only the
current session. The bar is not drawn when output goes to a file (or with
`--no-progress`); the line printed after each chunk still gives the time left.

`--chunks 0-31` runs only part of the chunk range, for example to spread a long
search over several sessions.

The merge step exits with an error unless the chunks cover the whole search space,
every candidate subset was scored successfully, all chunks were computed from the
given statistics file, and the stored scores match it. Only a passing merge counts as
an exact optimum. It warns if the list of near-ties may be incomplete, in which case
rerun the search with a larger `--top` in a new output directory.

## Timing

Every finished chunk appends a row to a timing log: k, band count, chunk, subsets,
seconds, subsets/s and device. By default the log is `timings.csv` next to the
`--out` directory, so `results/exact/k3`, `results/exact/k4` and so on share
`results/exact/timings.csv` (set another path with `--timings`). Each run also prints
its own total, and the merge step prints the summed compute time of all chunks for
that k. To compare band counts:

```bash
python scripts/timing_report.py results/exact/timings.csv --csv results/timing_summary.csv
```

The report gives one row per k, band count and device: chunks and subsets timed,
total compute time, subsets/s, and the projected time of the full search. A rerun
chunk counts once (its latest timing), and folds are pooled. The first chunk of each
run includes start-up work, so short runs read slightly slow.

For the RQ4 thread-scaling comparison, `benchmark.py --log results/benchmark.csv`
keeps every benchmark measurement as well.

## Layout

| Path | Purpose |
| --- | --- |
| `src/bandopt/sensors.py` | Band simulation (Eq. 1), Gaussian responses (Eq. 4), water-vapour mask |
| `src/bandopt/emit.py` | Read EMIT radiance, geometry, masks, and the lat/lon lookup grid |
| `src/bandopt/labels.py` | WorldCover purity, cloud polygons, and class labels |
| `src/bandopt/toa.py` | Radiance to TOA reflectance, and band-averaged solar irradiance |
| `src/bandopt/stats.py` | Class means, shrunk covariances, and scene-based folds |
| `src/bandopt/separability.py` | Batched Bhattacharyya, Jeffries–Matusita, and worst-case score (Eqs. 2–3), with unrolled small-matrix Cholesky |
| `src/bandopt/combinatorics.py` | Integer-to-subset mapping and chunking of the search space |
| `src/bandopt/exact.py` | Exhaustive scoring of one chunk, keeping the top subsets; device selection |
| `src/bandopt/timing.py` | Timing logs and their per-k summary |
| `scripts/` | Command-line entry points, including `benchmark.py` |
| `configs/` | Scene lists for `build_pixel_table.py` |
| `paper/` | LaTeX paper |
| `data/` | Local data, not committed. See [data/README.md](data/README.md) |
