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
| Ice-cloud samples | Polygon and L2A cloud flag | Polygon and the L2A cloud *or* cirrus flag (liquid still needs the cloud flag) | The cloud flag misses most ice anvils that are not very bright. The cirrus flag comes from the excluded 1.38 µm band |
| Folds | 5 | 3 | Ice cloud is in only two scenes; with 3 folds every fold's training scenes keep all classes |

"Exact optimum" means the exact optimum of the filter score on the training
statistics. Classifier results may rank subsets differently.

Search folds: every fold for k ≤ 4, one fixed scene split for k = 5, and a final run
on all data (`stats_all.npz`) only to report the bands.

Search space with n = 256 candidate bands (EMIT's 285 minus 29 in the water-vapour
windows):

| k | Subsets |
| --- | --- |
| 3 | 2.8×10⁶ |
| 4 | 1.7×10⁸ |
| 5 | 8.8×10⁹ |
| 6 | 3.7×10¹¹ |

On Windows, `environment.yml` pins NumPy to OpenBLAS: conda-forge's MKL clashes with
the DLLs bundled in the pip PyTorch wheel, and `import torch` fails with WinError 127.

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

## Downloading the data

[configs/scenes.toml](configs/scenes.toml) lists the 15 scenes used, with each one's
site, date, catalogue cloud cover and role. `scripts/download_data.py` fetches the
three EMIT files of every scene, the WorldCover tiles that overlap them, and the
TSIS-1 solar spectrum (about 30 GB). It needs an Earthdata login saved in `~/.netrc`
or `~/_netrc` (see [data/README.md](data/README.md)), skips files already downloaded,
and retries server errors, so an interrupted download resumes when run again:

```bash
python scripts/download_data.py --config configs/scenes.toml --dry-run   # what would be fetched
python scripts/download_data.py --config configs/scenes.toml
```

On the PC used so far `data/raw` is a directory junction to `D:\bandopt-data\raw`,
because C: is nearly full.

## Drawing cloud polygons

`scripts/draw_clouds.py` opens one scene on its map grid and saves liquid- and
ice-cloud polygons as the GeoJSON that the pixel table reads:

```bash
python scripts/draw_clouds.py --config configs/scenes.toml --scene india_storm
```

The default view is a cloud-phase composite (red 0.65 µm, green 1.6 µm, blue 2.2 µm):
liquid cloud looks white and ice looks red. Left-click corners, then press `l`
(liquid) or `i` (ice); `v` switches to true colour, `m` toggles the L2A cloud (cyan)
and cirrus (magenta) outlines, and `s` saves. The help at the top of the script lists
every key. Saved polygons load again next time, so you can review or edit them.

The polygons in `data/labels/` were drafted with AI assistance and need the author's
review before final runs.

## Building the pixel table

`scripts/build_pixel_table.py` turns EMIT granules into the labelled TOA reflectance
table that `compute_stats.py` reads. Copy `configs/scenes.example.toml` to
`configs/scenes.toml` and list each granule's three files (L1B radiance, L1B
observation geometry, L2A mask), then:

```bash
python scripts/build_pixel_table.py --config configs/scenes.toml --out data/processed/pixels.npz
python scripts/compute_stats.py --pixels data/processed/pixels.npz --out data/processed/stats --folds 3
```

Try one scene first with `--scenes <id>`; `compute_stats.py --folds 0` then writes only
`stats_all.npz`, since one scene cannot be split into folds. Besides the table, the
script writes `pixels.counts.csv` with pixel counts per scene and class.

How pixels are labelled:

- **Surface classes** come from WorldCover. A pixel needs at least 90% of its area in
  one class (`--purity`) and no L2A cloud, dilated cloud, cirrus, or spacecraft flag.
  Shrubland and cropland are left out by default. WorldCover has one bare class, so
  each scene's config says whether it is `bare_soil` or `sand`.
- **Cloud classes** come from polygons drawn with `draw_clouds.py` (or any GIS), saved
  as GeoJSON in longitude/latitude with a `label` property of `liquid_cloud` or
  `ice_cloud`. A liquid pixel must be inside a liquid polygon and carry the L2A cloud
  flag; an ice pixel must be inside an ice polygon and carry the cloud or the cirrus
  flag.
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

## Heuristics, OLI, classifiers and learned bands

Each script writes a CSV that `make_figures.py` turns into a table or figure.

```bash
# RQ2: SFS, SFFS, mRMR once; simulated annealing and a genetic algorithm per seed;
# group-lasso logistic regression (GL) when --pixels is given.
# Gaps are against the exact optimum where results/exact_all/k<k>/ exists.
python scripts/run_heuristics.py --stats data/processed/stats/stats_all.npz \
    --pixels data/processed/pixels.npz \
    --k 3 4 5 6 8 10 --seeds 30 --exact-dir results/exact_all --out results/heuristics_all.csv

# RQ1: simulated OLI bands, scored on the same statistics as the exact search
python scripts/evaluate_sensors.py --stats data/processed/stats/stats_all.npz \
    --exact-dir results/exact_all --out results/sensors_all.csv

# Wrapper check: LDA, random forest and MLP on held-out scenes, per fold
python scripts/evaluate_classifiers.py --pixels data/processed/pixels.npz \
    --stats-dir data/processed/stats --results-dir results --k 3 4 --out results/classifiers.csv

# RQ3: learnable Gaussian bands + classifier, per fold, from several random starts
python scripts/train_differentiable.py --pixels data/processed/pixels.npz \
    --stats-dir data/processed/stats --k 3 4 5 --seeds 3 --out results/differentiable.csv
```

OLI bands are Gaussians on the published band edges, not the official spectral
responses (yet). The learned-band absorption penalty uses a proxy that is 1 inside
the water-vapour windows. `build_pixel_table.py --screen-water 0.05
--screen-vegetation 0.4` optionally drops water pixels bright at 1.65 µm and
vegetation pixels with low NDVI.

## Running everything on the GPU workstation

The raw EMIT data and the pixel table are built on the data PC; the searches and
experiments run on the RTX 5080 workstation. Only the processed folder moves
(about 120 MB):

1. On the data PC, finish the inputs: review the cloud polygons, then build the
   screened table and statistics (or reuse `D:\bandopt-data\screened\processed`):
   ```bash
   python scripts/build_pixel_table.py --config configs/scenes.toml \
       --out D:/bandopt-data/screened/processed/pixels.npz --screen-water 0.05 --screen-vegetation 0.4
   python scripts/compute_stats.py --pixels D:/bandopt-data/screened/processed/pixels.npz \
       --out D:/bandopt-data/screened/processed/stats --folds 3
   ```
2. Copy `<data>/processed/` (`pixels.npz` and `stats/`) to the workstation.
3. On the workstation: `git pull`, then `conda env update -f environment.yml --prune`
   (or `conda env create` the first time), and check that PyTorch sees the GPU.
4. Run every step, which resumes if interrupted:
   ```bash
   python scripts/run_all.py --data <data> --device cuda
   ```
   This runs the exact searches (k = 3, 4 on all scenes and every fold; k = 5 on
   fold0), the OLI comparison, the heuristics, the classifier check, the learned
   bands, and `make_figures.py`. Commit `paper/figures` and `paper/generated` from
   there, or copy `<data>/results/` back and rerun `make_figures.py`.

## Paper figures and tables

`scripts/make_figures.py` writes every figure (`paper/figures/*.pdf`) and generated
table and number (`paper/generated/*.tex`) in the paper from the pipeline outputs.
Search results are read from `<results-dir>/exact_<stats name>/k<k>/`, so name
search output folders that way (`exact_all`, `exact_fold0`, ...). Rerun it after new
results, then rebuild the PDF:

```bash
python scripts/make_figures.py --pixels data/processed/pixels.npz \
    --stats-dir data/processed/stats --results-dir results
cd paper && latexmk -pdf main.tex
```

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
| `src/bandopt/progress.py` | Terminal progress bar for the exact search |
| `src/bandopt/heuristics.py` | SFS, SFFS, mRMR, simulated annealing, genetic algorithm |
| `src/bandopt/group_lasso.py` | Group-lasso multinomial logistic regression (FISTA path) |
| `src/bandopt/differentiable.py` | Learnable Gaussian bands, classifier, and training loop |
| `scripts/` | Command-line entry points: downloading, drawing cloud polygons, the pixel table, statistics, search, merge, timing, figures, and `benchmark.py` |
| `data/labels/` | Cloud polygons (GeoJSON), one file per cloud scene |
| `configs/` | Scene lists for `build_pixel_table.py` |
| `paper/` | LaTeX paper |
| `data/` | Local data, not committed. See [data/README.md](data/README.md) |
