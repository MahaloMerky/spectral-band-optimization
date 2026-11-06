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
(classes x bands x bands), and `wavelengths_nm`. Real statistics do not exist yet,
so use synthetic ones to try the pipeline:

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
STATS=data/processed/synthetic_stats.npz K=3 sbatch slurm/exact_search.sbatch
```

The merge step exits with an error unless the chunks cover the whole search space and
every candidate subset was scored successfully. Only a passing merge counts as an
exact optimum.

## Layout

| Path | Purpose |
| --- | --- |
| `src/bandopt/sensors.py` | Band simulation (Eq. 1), Gaussian responses (Eq. 4), water-vapour mask |
| `src/bandopt/stats.py` | Class means and shrunk covariances |
| `src/bandopt/separability.py` | Batched Bhattacharyya, Jeffries–Matusita, and worst-case score (Eqs. 2–3) |
| `src/bandopt/combinatorics.py` | Integer-to-subset mapping and chunking for job arrays |
| `src/bandopt/exact.py` | Exhaustive scoring of one chunk, keeping the top subsets |
| `scripts/` | Command-line entry points |
| `slurm/` | Job array templates |
| `data/` | Local data, not committed. See [data/README.md](data/README.md) |

## Roadmap

- [x] Core scoring, exact search chunks, tests
- [ ] **Week 1 (Nov 2):** download and tile EMIT scenes, build labels, compute class statistics, set up the cluster environment
- [ ] **Week 2 (Nov 9):** simulate OLI and Landsat Next, benchmark GPU throughput, exact search for k = 3, 4
- [ ] **Week 3 (Nov 16):** exact search for k = 5; greedy, floating search, mRMR, L1 methods
- [ ] **Week 4 (Nov 23):** genetic algorithm, simulated annealing, optimality-gap analysis
- [ ] **Week 5 (Nov 30):** differentiable design, constraint sweeps, classifier evaluation with spatial CV
- [ ] **Week 6 (Dec 7):** scaling study, robustness tests, stretch goals
- [ ] **Week 7 (Dec 14):** figures, physical interpretation, report, presentation
