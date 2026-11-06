# Optimizing Satellite Spectral Bands

GGS 426 (Fall 2026) final project: exact, heuristic, and differentiable band
selection for land-cover and cloud discrimination, using NASA EMIT hyperspectral
imagery on a SLURM cluster. See the [proposal](docs/GGS426_Final_Project_Proposal.pdf).

## Research questions

1. **RQ1:** Which k EMIT bands maximise class separability, and how far below that
   optimum are Landsat 8/9 OLI and Landsat Next?
2. **RQ2:** How close do heuristics come to the exact optimum, and how do they scale?
3. **RQ3:** Where does gradient-based sensor design place band centres and widths,
   and is the placement physically interpretable?
4. **RQ4:** How do run time and parallel efficiency scale with cores and GPUs?

## Changes from the proposal

| Topic | Proposal | Now | Reason |
| --- | --- | --- | --- |
| Input data | L2A surface reflectance | L1B radiance converted to TOA reflectance, for all classes | Surface reflectance retrieval assumes a view of the surface, which does not hold over clouds. One quantity keeps all classes comparable |
| Search score | Worst-case JM | Worst-case Bhattacharyya distance, reported as JM | Same ranking, without JM's rounding ties close to 2 |
| Near-ties | Not specified | Subsets within 0.001 JM of the best are ranked by mean JM | Gives a defined choice among practically equal subsets |
| Leakage | Spatial CV for evaluation | Class statistics and band selection also use training scenes only | Selecting bands with held-out scenes would inflate scores |
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
```

After the environment works on the cluster, record exact versions so runs are
reproducible, and commit the result:

```bash
conda env export --no-builds > environment.lock.yml
```

If the cluster needs a specific CUDA build of PyTorch, install it with the
[selector command](https://pytorch.org/get-started/locally/) before exporting.

## Exact search

The search reads a statistics file holding `mu` (classes x bands), `sigma`
(classes x bands x bands), and `wavelengths_nm`. For real data,
`scripts/compute_stats.py` writes one file per scene-based fold from a table of
labelled TOA pixels (add `--cirrus-run` for the sensitivity run). That table does not
exist yet, so use synthetic statistics to try the pipeline:

```bash
python scripts/make_synthetic_stats.py --out data/processed/synthetic_stats.npz

# Run every chunk locally
for t in 0 1 2 3; do
  python scripts/exact_search.py --stats data/processed/synthetic_stats.npz --k 3 \
    --task-id $t --n-tasks 4 --out results/exact/k3
done
python scripts/merge_exact.py results/exact/k3 --stats data/processed/synthetic_stats.npz

# Or on the cluster: set the partition in the script first
mkdir -p logs
STATS=data/processed/synthetic_stats.npz K=3 OUT=results/exact/k3 sbatch slurm/exact_search.sbatch
```

The merge step exits with an error unless the chunks cover the whole search space,
every candidate subset was scored successfully, and the scores match the statistics
file. Only a passing merge counts as an exact optimum. It warns if the list of
near-ties may be incomplete, in which case rerun the search with a larger `--top`.

## Layout

| Path | Purpose |
| --- | --- |
| `src/bandopt/sensors.py` | Band simulation (Eq. 1), Gaussian responses (Eq. 4), water-vapour mask |
| `src/bandopt/toa.py` | Radiance to TOA reflectance, and band-averaged solar irradiance |
| `src/bandopt/stats.py` | Class means, shrunk covariances, and scene-based folds |
| `src/bandopt/separability.py` | Batched Bhattacharyya, Jeffries–Matusita, and worst-case score (Eqs. 2–3) |
| `src/bandopt/combinatorics.py` | Integer-to-subset mapping and chunking for job arrays |
| `src/bandopt/exact.py` | Exhaustive scoring of one chunk, keeping the top subsets |
| `scripts/` | Command-line entry points |
| `slurm/` | Job array templates |
| `data/` | Local data, not committed. See [data/README.md](data/README.md) |
