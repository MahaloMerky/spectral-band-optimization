"""Heuristic band selection for the worst-case separability score.

Every method maximises the same objective as the exact search: the smallest
pairwise Bhattacharyya distance over class pairs (equivalent to worst-case JM).
They return a sorted list of band indices and its score.

- sequential_forward_selection (SFS): add the band that helps most, k times.
- sequential_floating_forward_selection (SFFS): SFS with conditional removal
  steps (Pudil et al. 1994).
- mrmr: minimum redundancy, maximum relevance (Peng et al. 2005), adapted to
  class statistics: relevance is a band's own worst-case distance, redundancy
  its mean absolute pooled within-class correlation with the bands chosen.
- simulated_annealing: random single-band swaps, accepted by the Metropolis rule.
- genetic_algorithm: tournament selection, set crossover, swap mutation, elitism.
"""

import math

import numpy as np
import torch

from bandopt.separability import worst_case_bhattacharyya


class Scorer:
    """Score subsets by worst-case Bhattacharyya distance, counting evaluations."""

    def __init__(self, mu, sigma, batch_size=20_000):
        self.mu = torch.as_tensor(mu, dtype=torch.float64)
        self.sigma = torch.as_tensor(sigma, dtype=torch.float64)
        self.n_bands = self.mu.shape[1]
        self.batch_size = batch_size
        self.evaluations = 0

    @torch.inference_mode()
    def __call__(self, subsets):
        """Scores for a list or array of subsets of equal size; failures score -inf."""
        subsets = torch.as_tensor(np.asarray(subsets), dtype=torch.int64)
        if subsets.ndim == 1:
            subsets = subsets[None]
        self.evaluations += len(subsets)
        scores = []
        for start in range(0, len(subsets), self.batch_size):
            chunk = torch.sort(subsets[start:start + self.batch_size], dim=1).values
            scores.append(worst_case_bhattacharyya(self.mu, self.sigma, chunk))
        scores = torch.cat(scores)
        return torch.nan_to_num(scores, nan=-math.inf).numpy()


def _best_addition(scorer, selected, candidates):
    trials = np.array([sorted(selected + [b]) for b in candidates])
    scores = scorer(trials)
    best = int(np.argmax(scores))
    return candidates[best], float(scores[best])


def sequential_forward_selection(scorer, k):
    selected, score = [], -math.inf
    for _ in range(k):
        remaining = [b for b in range(scorer.n_bands) if b not in selected]
        band, score = _best_addition(scorer, selected, remaining)
        selected.append(band)
    return sorted(selected), score


def sequential_floating_forward_selection(scorer, k, max_steps=1000):
    """SFFS: after each addition, remove bands while that beats the best subset
    found so far of the smaller size."""
    selected = []
    best = {}  # size -> (score, subset)
    for _ in range(max_steps):
        if len(selected) == k:
            break
        remaining = [b for b in range(scorer.n_bands) if b not in selected]
        band, score = _best_addition(scorer, selected, remaining)
        selected = sorted(selected + [band])
        if score > best.get(len(selected), (-math.inf,))[0]:
            best[len(selected)] = (score, selected)
        while len(selected) > 2:
            trials = np.array([[b for b in selected if b != drop] for drop in selected])
            scores = scorer(trials)
            i = int(np.argmax(scores))
            if scores[i] > best.get(len(selected) - 1, (-math.inf,))[0] + 1e-12:
                selected = trials[i].tolist()
                best[len(selected)] = (float(scores[i]), selected)
            else:
                break
    while len(selected) < k:  # step limit reached: finish greedily
        remaining = [b for b in range(scorer.n_bands) if b not in selected]
        band, score = _best_addition(scorer, selected, remaining)
        selected = sorted(selected + [band])
        best[len(selected)] = max(best.get(len(selected), (-math.inf, None)), (score, selected),
                                  key=lambda item: item[0])
    score, subset = best[k]
    return sorted(subset), score


def pooled_correlation(sigma):
    """Mean within-class correlation matrix from class covariances."""
    sigma = np.asarray(sigma).mean(axis=0)
    sd = np.sqrt(np.diag(sigma))
    return sigma / np.outer(sd, sd)


def mrmr(scorer, k):
    single = scorer(np.arange(scorer.n_bands)[:, None])
    relevance = (single - single.min()) / (single.max() - single.min() + 1e-12)
    correlation = np.abs(pooled_correlation(scorer.sigma.numpy()))
    selected = [int(np.argmax(relevance))]
    while len(selected) < k:
        redundancy = correlation[:, selected].mean(axis=1)
        value = relevance - redundancy
        value[selected] = -np.inf
        selected.append(int(np.argmax(value)))
    selected = sorted(selected)
    return selected, float(scorer(np.array([selected]))[0])


def _swap(current, n, rng):
    """Replace one random band of each row with a band that row does not contain."""
    candidate = current.copy()
    new_band = rng.integers(n, size=len(current))
    clash = (candidate == new_band[:, None]).any(axis=1)
    while clash.any():
        new_band[clash] = rng.integers(n, size=int(clash.sum()))
        clash = (candidate == new_band[:, None]).any(axis=1)
    candidate[np.arange(len(current)), rng.integers(current.shape[1], size=len(current))] = new_band
    candidate.sort(axis=1)
    return candidate


def simulated_annealing(scorer, k, rng, iterations=20_000, chains=1, start_temperature=None,
                        cooling=1e-3, calibration_moves=256):
    """Run independent annealing chains in lockstep, scoring all chains in one batch.

    Each step swaps one band of every chain for a band it does not contain and
    accepts by the Metropolis rule. The temperature falls geometrically from
    start_temperature to start_temperature * cooling. By default the start is
    calibrated so that a typical (median) worsening move from random subsets is
    accepted half the time. Returns a list of (subset, score), one per chain;
    with chains=1 a single tuple.
    """
    n = scorer.n_bands
    current = np.array([np.sort(rng.choice(n, k, replace=False)) for _ in range(chains)])
    current_score = scorer(current)
    best, best_score = current.copy(), current_score.copy()
    if start_temperature is None:
        probes = np.array([np.sort(rng.choice(n, k, replace=False)) for _ in range(calibration_moves)])
        delta = scorer(_swap(probes, n, rng)) - scorer(probes)
        worse = -delta[np.isfinite(delta) & (delta < 0)]
        start_temperature = float(np.median(worse)) / math.log(2) if worse.size else 1.0
    end_temperature = start_temperature * cooling
    for step in range(iterations):
        temperature = start_temperature * (end_temperature / start_temperature) ** (step / iterations)
        candidate = _swap(current, n, rng)
        score = scorer(candidate)
        with np.errstate(over="ignore", invalid="ignore"):
            accept = (score >= current_score) | (
                rng.random(chains) < np.exp(np.minimum((score - current_score) / temperature, 0)))
        current[accept], current_score[accept] = candidate[accept], score[accept]
        better = current_score > best_score
        best[better], best_score[better] = current[better], current_score[better]
    results = [(best[i].tolist(), float(best_score[i])) for i in range(chains)]
    return results[0] if chains == 1 else results


def _repair(subset, k, n, rng):
    subset = list(dict.fromkeys(subset))
    while len(subset) < k:
        band = int(rng.integers(n))
        if band not in subset:
            subset.append(band)
    return sorted(subset[:k])


def genetic_algorithm(scorer, k, rng, population=200, generations=150, mutation=0.3, elite=4, tournament=3):
    n = scorer.n_bands
    pop = np.array([sorted(rng.choice(n, k, replace=False)) for _ in range(population)])
    fitness = scorer(pop)
    for _ in range(generations):
        order = np.argsort(-fitness)
        children = [pop[i].tolist() for i in order[:elite]]
        while len(children) < population:
            parents = []
            for _ in range(2):
                entrants = rng.choice(population, tournament, replace=False)
                parents.append(pop[entrants[np.argmax(fitness[entrants])]])
            union = np.union1d(*parents)
            child = rng.choice(union, k, replace=False).tolist() if len(union) >= k else union.tolist()
            if rng.random() < mutation:
                child[int(rng.integers(len(child)))] = int(rng.integers(n))
            children.append(_repair(child, k, n, rng))
        pop = np.array(children)
        fitness = scorer(pop)
    best = int(np.argmax(fitness))
    return pop[best].tolist(), float(fitness[best])


DETERMINISTIC = {
    "SFS": sequential_forward_selection,
    "SFFS": sequential_floating_forward_selection,
    "mRMR": mrmr,
}
STOCHASTIC = {
    "SA": simulated_annealing,
    "GA": genetic_algorithm,
}
