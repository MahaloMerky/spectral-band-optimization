"""Tests for the heuristics, sensor simulation, screening, and differentiable design."""

import itertools

import numpy as np
import pytest
import torch

from bandopt.differentiable import LearnableSensor, predict, train, window_absorption
from bandopt.heuristics import (
    Scorer,
    genetic_algorithm,
    mrmr,
    sequential_floating_forward_selection,
    sequential_forward_selection,
    simulated_annealing,
)
from bandopt.group_lasso import GroupLassoPath, group_soft_threshold, select_bands
from bandopt.labels import class_index, spectral_screen
from bandopt.sensors import OLI_BAND_EDGES_NM, OLI_REFLECTIVE, band_weights


def random_statistics(n_classes=4, n_bands=12, seed=0):
    generator = torch.Generator().manual_seed(seed)
    mu = torch.rand(n_classes, n_bands, generator=generator, dtype=torch.float64)
    factors = torch.randn(n_classes, n_bands, n_bands, generator=generator, dtype=torch.float64)
    sigma = factors @ factors.mT / n_bands + 0.05 * torch.eye(n_bands, dtype=torch.float64)
    return mu, sigma


def brute_force(scorer, k):
    subsets = np.array(list(itertools.combinations(range(scorer.n_bands), k)))
    scores = scorer(subsets)
    return float(scores.max())


@pytest.fixture
def scorer():
    return Scorer(*random_statistics())


@pytest.mark.parametrize("k", [2, 3])
def test_stochastic_heuristics_find_the_brute_force_optimum(scorer, k):
    optimum = brute_force(scorer, k)
    _, ga = genetic_algorithm(scorer, k, np.random.default_rng(0), population=60, generations=40)
    # Independent chains can settle in a near-optimal local optimum; the best of
    # several should reach the global one.
    chains = simulated_annealing(scorer, k, np.random.default_rng(0), iterations=5000, chains=8)
    assert ga == pytest.approx(optimum)
    assert max(score for _, score in chains) == pytest.approx(optimum)


def test_greedy_heuristics_return_valid_subsets(scorer):
    k = 4
    optimum = brute_force(scorer, k)
    sfs_subset, sfs = sequential_forward_selection(scorer, k)
    sffs_subset, sffs = sequential_floating_forward_selection(scorer, k)
    mrmr_subset, mrmr_score = mrmr(scorer, k)
    for subset, score in ((sfs_subset, sfs), (sffs_subset, sffs), (mrmr_subset, mrmr_score)):
        assert len(set(subset)) == k and subset == sorted(subset)
        assert score <= optimum + 1e-12
        assert score == pytest.approx(float(scorer(np.array([subset]))[0]))
    assert sffs >= sfs - 1e-12


def test_simulated_annealing_single_chain_returns_one_result(scorer):
    subset, score = simulated_annealing(scorer, 3, np.random.default_rng(1), iterations=200)
    assert len(subset) == 3 and np.isfinite(score)


def test_band_weights_are_normalised_and_centred():
    grid = np.arange(400.0, 2500.0, 7.4)
    edges = [OLI_BAND_EDGES_NM[name] for name in OLI_REFLECTIVE]
    weights = band_weights(grid, edges).numpy()
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)
    centres = weights @ grid
    np.testing.assert_allclose(centres, [np.mean(e) for e in edges], atol=4.0)


def test_simulated_band_statistics_match_simulated_pixels():
    rng = np.random.default_rng(0)
    grid = np.linspace(400, 2500, 60)
    pixels = rng.random((500, grid.size))
    weights = band_weights(grid, [(600, 700), (800, 900)]).numpy()
    bands = pixels @ weights.T
    np.testing.assert_allclose(weights @ pixels.mean(axis=0), bands.mean(axis=0))
    np.testing.assert_allclose(weights @ np.cov(pixels.T) @ weights.T, np.cov(bands.T))


def test_spectral_screen_drops_only_failing_water_and_vegetation():
    water, vegetation, urban = class_index("water"), class_index("vegetation"), class_index("urban")
    labels = np.array([water, water, vegetation, vegetation, urban])
    red = np.array([0.05, 0.05, 0.05, 0.20, 0.20])
    nir = np.array([0.04, 0.04, 0.40, 0.22, 0.25])
    swir = np.array([0.02, 0.15, 0.20, 0.30, 0.40])
    keep = spectral_screen(labels, red, nir, swir, water_max_swir=0.05, vegetation_min_ndvi=0.4)
    assert keep.tolist() == [True, False, True, False, True]
    assert spectral_screen(labels, red, nir, swir).all()


def test_window_absorption_is_one_inside_and_zero_far_outside():
    grid = torch.tensor([1000.0, 1385.0, 1355.0, 1345.0, 1335.0, 1900.0, 2200.0])
    absorption = window_absorption(grid).tolist()
    assert absorption[0] == 0 and absorption[-1] == 0
    assert absorption[1] == 1 and absorption[2] == 1 and absorption[5] == 1
    assert 0 < absorption[3] < 1 and absorption[4] == 0


def test_learnable_sensor_gradients_reach_centres_and_widths():
    grid = np.linspace(400, 2500, 100)
    sensor = LearnableSensor(grid, 3, init_centres_nm=[500, 900, 1600], noise=0.0)
    spectra = torch.rand(8, grid.size)
    sensor(spectra).sum().backward()
    assert sensor.centre_logit.grad.abs().sum() > 0
    assert sensor.fwhm_raw.grad.abs().sum() > 0
    np.testing.assert_allclose(sensor.centres.detach().numpy(), [500, 900, 1600], rtol=1e-4)


def test_differentiable_training_learns_a_separable_toy_problem():
    rng = np.random.default_rng(0)
    grid = np.linspace(400, 2500, 120)
    # Two classes that differ only in the near infrared, as vegetation and soil do.
    labels = rng.integers(0, 2, 600)
    spectra = 0.3 + 0.02 * rng.standard_normal((600, grid.size))
    spectra[:, (grid > 700) & (grid < 1300)] += 0.3 * labels[:, None]
    # As in train_differentiable.py: several random starts, keep the lowest training loss.
    runs = [train(spectra, labels, grid, 2, 2, epochs=60, batch_size=128, noise=0.0, seed=seed)
            for seed in range(3)]
    model, history = min(runs, key=lambda run: run[1]["loss"][-1])
    accuracy = (predict(model, spectra) == labels).mean()
    assert accuracy > 0.95
    assert history["loss"][-1] < history["loss"][0]
    assert ((model.sensor.centres > 650) & (model.sensor.centres < 1350)).any()


def test_group_soft_threshold_zeroes_small_rows_and_shrinks_large_ones():
    weights = torch.tensor([[3.0, 4.0], [0.3, 0.4]], dtype=torch.float64)
    shrunk = group_soft_threshold(weights, 1.0)
    np.testing.assert_allclose(shrunk[0].numpy(), [2.4, 3.2])  # norm 5 -> 4
    np.testing.assert_allclose(shrunk[1].numpy(), [0.0, 0.0])


def test_group_lasso_selects_the_informative_bands():
    rng = np.random.default_rng(0)
    n, p = 1500, 20
    labels = rng.integers(0, 3, n)
    features = rng.standard_normal((n, p))
    features[:, 4] += 2.0 * (labels == 1)
    features[:, 11] += 2.0 * (labels == 2)
    chosen, penalty = select_bands(features, labels, 3, 2)
    assert chosen == [4, 11]
    assert penalty > 0


def test_group_lasso_has_no_bands_at_lambda_max():
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 2, 400)
    features = rng.standard_normal((400, 6)) + labels[:, None] * 0.5
    path = GroupLassoPath(features, labels, 2)
    assert path.fit(path.lambda_max() * 1.01) == []
    assert len(path.fit(path.lambda_max() * 0.05)) > 0
